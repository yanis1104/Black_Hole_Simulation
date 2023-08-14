#pragma once

#include <SFML/Graphics.hpp>
#include <cmath>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

std::vector<std::vector<std::vector<sf::ConvexShape>>> computeDisk(sf::RenderWindow& window);