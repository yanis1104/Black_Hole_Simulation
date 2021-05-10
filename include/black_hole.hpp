/*
    Yanis GOUGEAT
    May 2021

    Black Hole simulation
*/

#ifndef BLACK_HOLE_HPP_
#define BLACK_HOLE_HPP_

#include <cmath>
#include <cstddef>
#include <string>
#include <iostream>
#include <SFML/Window.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <SFML/System.hpp>
#include <SFML/Graphics/Export.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/PrimitiveType.hpp>

#define PI (3.14159265358979323846)

class ParticleSystem : public sf::Drawable, public sf::Transformable
{
    public:
        ParticleSystem(sf::Image &background);
        virtual ~ParticleSystem();
        virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
        sf::VertexArray vertex;
};

#endif
