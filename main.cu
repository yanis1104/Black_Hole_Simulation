#include <cuda.h>
#include <cuda_runtime.h>
#include "compute_black_hole.cuh"
#include "rotate_background.cuh"
#include "Disk.h"

void init_backgrounds(int2 dimensions, int3* background_absolute, int3* d_background_absolute, sf::Image &background)
{
    for (int i = 0, k = 0; i < dimensions.x; i++) {
        for (int j = 0; j < dimensions.y; j++, k++) {
            background_absolute[k].x = background.getPixel(i, j).r;
            background_absolute[k].y = background.getPixel(i, j).g;
            background_absolute[k].z = background.getPixel(i, j).b;
        }
    }
    cudaMemcpy(d_background_absolute, background_absolute, dimensions.x * dimensions.y * sizeof(int3), cudaMemcpyHostToDevice);
}

void move_blackhole(float2 &BH_pos, sf::RenderWindow &window, int2 dimensions)
{
    if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
        BH_pos.x = (float)sf::Mouse::getPosition(window).x;
        BH_pos.y = (float)sf::Mouse::getPosition(window).y;
        if (BH_pos.x < 0)
            BH_pos.x = 0;
        if (BH_pos.x > dimensions.x)
            BH_pos.x = (float)dimensions.x - 1;
        if (BH_pos.y < 0)
            BH_pos.y = 0;
        if (BH_pos.y > dimensions.y)
            BH_pos.y = (float)dimensions.y - 1;
    }
}

void draw_window(sf::RenderWindow &window, sf::Texture &texture, sf::Sprite &sprite, sf::Image &image, sf::Clock &clock)
{
    texture.loadFromImage(image);
    sprite.setTexture(texture);
    //window.clear();
    window.draw(sprite);
    //window.display();
    clock.restart();
}

void set_disk_texture(std::vector<std::vector<std::vector<sf::ConvexShape>>> &shapes, sf::Texture &disk_texture, int &line, sf::RenderWindow &window)
{
    for (int i = 0; i < shapes.size(); i++) {
        for (int j = 0; j < shapes[i].size(); j++) {
            for (int k = 0; k < shapes[i][j].size(); k++) {
                shapes[i][j][k].setTexture(&disk_texture);
                shapes[i][j][k].setTextureRect(sf::IntRect(line, 0, 50, 1000));
                shapes[i][j][k].setRotation(10.0);
                shapes[i][j][k].setPosition(sf::Vector2f(85, -135));
            }
        }
    }
}

void draw_disk(std::vector<std::vector<std::vector<sf::ConvexShape>>>& shapes, sf::RenderWindow &window)
{
    for (int i = 0; i < shapes.size(); i++) {
        for (int j = 0; j < shapes[i].size(); j++) {
            for (int k = 0; k < shapes[i][j].size(); k++) {
                window.draw(shapes[i][j][k]);
            }
        }
    }
}

void save_pictures(sf::RenderWindow& window)
{
    static int n = 0;
    sf::Texture texture;
    texture.create(window.getSize().x, window.getSize().y);
    texture.update(window);

    // Créez une image et copiez le contenu de la texture dans l'image
    sf::Image image = texture.copyToImage();

    // Enregistrez l'image dans un fichier
    image.saveToFile("results/image" + std::to_string(n) + ".png");
    n++;
}

int main()
{
    /////////////////

    sf::ContextSettings settings;
    settings.antialiasingLevel = 16.0;

    ////////////////
    sf::Image background;
    background.loadFromFile("include/background.png");
    sf::Texture texture;
    texture.loadFromImage(background);
    sf::Sprite sprite;
    sprite.setTexture(texture);
    sf::Image image = background;
    int2 dimensions = { (int)background.getSize().x, (int)background.getSize().y };
    sf::RenderWindow window(sf::VideoMode(dimensions.x, dimensions.y), "Black Hole Simulation", sf::Style::Close, settings);
    window.setFramerateLimit(30);
    sf::Clock clock;
    sf::Clock background_clock;

    dim3 block(16, 16);
    dim3 grid((dimensions.x + block.x - 1) / block.x, (dimensions.y + block.y - 1) / block.y);

    int3* background_absolute = new int3[dimensions.x * dimensions.y];
    int3* d_background_absolute;
    cudaMalloc((void**)&d_background_absolute, dimensions.x * dimensions.y * sizeof(int3));
    init_backgrounds(dimensions, background_absolute, d_background_absolute, background);
    
    int3* d_pixel_results;
    cudaMalloc((void**)&d_pixel_results, dimensions.x * dimensions.y * sizeof(int3));

    float2 BH_pos = { (float)dimensions.x / 2, (float)dimensions.y / 2 };

    //////////////////////////

    sf::Image disk_image;
    disk_image.loadFromFile("include/texture.jpg");

    sf::Texture disk_texture;
    disk_texture.loadFromImage(disk_image);

    sf::Sprite disk_sprite;
    disk_sprite.setTexture(disk_texture);

    std::vector<std::vector<std::vector<sf::ConvexShape>>> shapes = computeDisk(window);

    for (int i = 0; i < shapes.size(); i++) {
        for (int j = 0; j < shapes[i].size(); j++) {
            for (int k = 0; k < shapes[i][j].size(); k++) {
                shapes[i][j][k].setTexture(&disk_texture);
            }
        }
    }

    int line = 0;
    bool to_left = false;
    ////////////////////////////

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed || sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
                window.close();
            }
            //move_blackhole(BH_pos, window, dimensions);
        }

        compute_black_hole(dimensions, block, grid, d_pixel_results, d_background_absolute, BH_pos, image);

        shapes = computeDisk(window);
        if (clock.getElapsedTime().asMilliseconds() > 50) {
            window.clear();
            draw_window(window, texture, sprite, image, clock);
            set_disk_texture(shapes, disk_texture, line, window);
            draw_disk(shapes, window);
            window.display();
            //save_pictures(window);
            //ROTATION DISK
            if (!to_left) {
                if ((line += 40) >= 1000 - 50) {
                    line = 1000 - 50;
                    to_left = !to_left;
                }
            } else {
                if ((line -= 40) <= 0 + 50) {
                    line = 0;
                    to_left = !to_left;
                }
            }
            
        }
        move_background_to_left(d_background_absolute, image, background_clock, dimensions);
        
    }
    cudaFree(d_pixel_results);
    cudaFree(d_background_absolute);
    return 0;
}