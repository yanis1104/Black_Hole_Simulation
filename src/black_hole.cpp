/*
    Yanis GOUGEAT
    May 2021

    Black Hole simulation
*/

#include "black_hole.hpp"

void ParticleSystem::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
    states.transform *= getTransform();
    states.texture = NULL;
    target.draw(vertex, states);
}

ParticleSystem::ParticleSystem(sf::Image &background) : vertex(sf::Points, background.getSize().x * background.getSize().y)
{
    for (unsigned int x = 0, i = 0; x < background.getSize().x; x++) {
        for (unsigned int y = 0; y < background.getSize().y; y++) {
            this->vertex[i].color = background.getPixel(x, y);
            this->vertex[i].position.x = x;
            this->vertex[i].position.y = y;
            i++;
        }
    }
}

ParticleSystem::~ParticleSystem()
{
}