/*
    Yanis GOUGEAT
    May 2021

    Black Hole simulation
*/

#include "black_hole.hpp"

void create_black_hole(sf::Image &background, ParticleSystem &pixel, float bhX, float bhY)
{
    float width = background.getSize().x;
    float height = background.getSize().y;

    float length = 5.0;
    float radius = 0.1;
    float dt = 0.1;
    float maxDiastance = 1.00001;

    float n = 0.0;
    float radDistL = 0.0;
    float iPhoton = 0.0;
    float jPhoton = 0.0;

    float alpha = 0.0;
    float beta = 0.0;
    float x = 1.0;
    float y = 0.0;
    float t = 0.0;

    float vx = 0.0;
    float vy = 0.0;
    float ax = 0.0;
    float ay = 0.0;

    float dphi = 0.0;
    float pAlpha = 0.0;

    for (unsigned int i = 0; i < width; i++) {
        for (unsigned int j = 0; j < height; j++) {
            alpha = sqrt(pow(i - bhX, 2) + pow(bhY - j, 2)) * length / (width);
            beta = atan2(bhY - j, i - bhX);
            x = 1.0;
            y = 0.0;
            t = 0.0;
            vx = -cos(alpha);
            vy = sin(alpha);
            while (sqrt(x*x + y*y) < maxDiastance && sqrt(x*x + y*y) > radius) {
                t += dt;
                x += dt * vx;
                y += dt * vy;
                radDistL = sqrt(x*x + y*y);
                dphi = (x * vy - y * vx) / radDistL / radDistL;
                ax = (-3 / 2) * radius * (dphi * dphi) * x / radDistL;
                ay = (-3 / 2) * radius * (dphi * dphi) * y / radDistL;
                vx += dt * ax;
                vy += dt * ay;
            }
            if (radDistL >= maxDiastance) {
                pAlpha = atan2(y, 1 - x);
                iPhoton = bhX + pAlpha * cos(beta) * width / PI / 2;
                jPhoton = bhY - pAlpha * sin(beta) * height / PI;

                iPhoton = round(iPhoton);
                jPhoton = round(jPhoton);

                while (iPhoton < 0)
                    iPhoton = iPhoton + width;
                while (iPhoton >= width)
                    iPhoton = iPhoton - width;
                while (jPhoton < 0 )
                    jPhoton = jPhoton + height;
                while (jPhoton >= height)
                    jPhoton = jPhoton - height;
                pixel.vertex[n].color = background.getPixel(iPhoton, jPhoton);
            } else
                pixel.vertex[n].color = sf::Color(0, 0, 0);
            n++;
        }
    }
}

int main()
{
    sf::Image background;
    background.loadFromFile("include/background.png");
    sf::Image res;
    ParticleSystem pixel(background);
    int n = 0;

    std::cout << "SIMULATION STARTING..." << std::endl << std::endl;
    while (1) {
        create_black_hole(background, pixel, 2 * background.getSize().x / 3 - n, background.getSize().y / 2);
        res.create(background.getSize().x, background.getSize().y);
        for (unsigned int x = 0, i = 0; x < background.getSize().x; x++) {
            for (unsigned int y = 0; y < background.getSize().y; y++) {
                res.setPixel(x, y, pixel.vertex[i].color);
                i++;
            }
        }
        res.saveToFile("result/res" + std::to_string(n) + ".png");
        n += 1;
        std::cout << "Picture " << n << " of 375 created" << std::endl;
        if (n >= 750)
            break;   
    }
    std::cout << "SIMULATION FINISHED" << std::endl;
    return 0;
}