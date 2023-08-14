#include <SFML/Window.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <SFML/System.hpp>
#include <SFML/Graphics/Export.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/PrimitiveType.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

void move_background_to_left(int3* d_background_absolute, sf::Image& image, sf::Clock& background_clock, int2 dimensions);