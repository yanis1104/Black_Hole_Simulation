#include "Disk.h"

const float PI = 3.14;
const int window_size_X = 1600;
const int window_size_Y = 800;

struct Vec4 {
    double x;
    double y;
    double z;
    double w;
};

double impactParameterApprox(double spannedAngle, double camD) {

    double K = 2.27 / camD;
    double J = -0.162 / pow(camD, 0.297);
    double N = 2.81 * pow(camD, 0.4325);
    double O = pow(pow(spannedAngle - J, N) + pow(PI, N), 1 / N) + K;
    double res = 1.0 - exp(PI - O) + 2.1 * exp(-0.8 * O) * (1.0 - exp(1.67 * (PI - O)));
    double b = 3 * sqrt(3.0) / 2.0 / res;
    return b;
}

double acot(double x) {
    return PI / 2.0 - atan(x);
}

double cot(double x) {
    return 1.0 / tan(x);
}

int sign(double n)
{
    if (n > 0)
        return 1;
    else if (n < 0)
        return -1;
    else
        return 0;
}

double BfromRAlpha(double &r, double &alpha) {

    bool filter = (r > 1.85) && (r < 100);
    bool visible = false;
    double value = 2.598;

    if (filter) {
        double O = 1.53 / pow(r - 1.5, 0.19) + 0.059 * exp(pow(r + 0.1, 0.716)) - 0.0582;
        double M = 0.43 * exp(-0.227 * pow(r - 1.5, 1.28));
        double b1 = 0.5 + 0.37 / (r - 0.88) + r;
        double X = pow(b1 / r / O, 2);
        double alphaMax = O * acot(sqrt((4.0 / PI / PI - M) / X));

        if (alpha > 0) {
            visible = true;
            if (alpha < alphaMax) {
                value = b1 * sin(1.0 / sqrt(X * pow(cot(alpha / O), 2.0) + M));

            }
            else {
                double L = 0.36 * exp(0.11 * pow(log(r - 1.5) - 1.2, 2.0));
                value = 2.598 + (b1 - 2.598) * exp(-L * pow(alpha - alphaMax, 2.0));
            }
            return value;
        }
    }
    return 0;
}

std::vector<sf::ConvexShape> textureMap(std::vector<Vec4> &pts, double expand, double uOffset, sf::RenderWindow& window)
{
    sf::ConvexShape convex(3);
    std::vector<sf::ConvexShape> shapes;
    int discTexWidth = 1000;
    int discTexHeight = 1000;
    std::vector<sf::Vector3i> tris = { {0, 1, 2} ,{2, 3, 0} };

    for (int t = 0; t < 2; t++) {
        sf::Vector3i pp = tris[t];
        double x0 = pts[pp.x].x, x1 = pts[pp.y].x, x2 = pts[pp.z].x;
        double y0 = pts[pp.x].y, y1 = pts[pp.y].y, y2 = pts[pp.z].y;
        double u0 = pts[pp.x].z + uOffset, u1 = pts[pp.y].z + uOffset, u2 = pts[pp.z].z + uOffset;
        double v0 = pts[pp.x].w, v1 = pts[pp.y].w, v2 = pts[pp.z].w;

        double minU = std::min(u0, std::min(u1, u2));
        double maxU = std::max(u0, std::max(u1, u2));
        double shiftU = -std::floor(minU / (double)discTexWidth);
        u0 += shiftU * (double)discTexWidth;
        u1 += shiftU * (double)discTexWidth;
        u2 += shiftU * (double)discTexWidth;
        u0 = std::min(u0, (double)discTexWidth);
        u1 = std::min(u1, (double)discTexWidth);
        u2 = std::min(u2, (double)discTexWidth);

        double delta = u0 * v1 + v0 * u2 + u1 * v2 - v1 * u2 - v0 * u1 - u0 * v2;
        double delta_a = x0 * v1 + v0 * x2 + x1 * v2 - v1 * x2 - v0 * x1 - x0 * v2;
        double delta_b = u0 * x1 + x0 * u2 + u1 * x2 - x1 * u2 - x0 * u1 - u0 * x2;
        double delta_c = u0 * v1 * x2 + v0 * x1 * u2 + x0 * u1 * v2 - x0 * v1 * u2 - v0 * u1 * x2 - u0 * x1 * v2;
        double delta_d = y0 * v1 + v0 * y2 + y1 * v2 - v1 * y2 - v0 * y1 - y0 * v2;
        double delta_e = u0 * y1 + y0 * u2 + u1 * y2 - y1 * u2 - y0 * u1 - u0 * y2;
        double delta_f = u0 * v1 * y2 + v0 * y1 * u2 + y0 * u1 * v2 - y0 * v1 * u2 - v0 * u1 * y2 - u0 * y1 * v2;

        x0 -= (x1 - x0) * expand;
        y0 -= (y1 - y0) * expand;

        
        convex.setPoint(0, sf::Vector2f(x0, y0));
        convex.setPoint(1, sf::Vector2f(x1, y1));
        convex.setPoint(2, sf::Vector2f(x2, y2));

        shapes.push_back(convex);

    }
    return shapes;
}

std::vector<std::vector<std::vector<sf::ConvexShape>>> computeDisk(sf::RenderWindow &window) {

    double discRadius = 15.0;
    double maxOrderDisc = 2.0;
    static double camPhi = 0.00;//1.2027967150414685;
    static double camTheta = -0.155;//-0.05019651085250125;
    double discQuality = 10000.0;
    double rIn = 3.0;
    double rOut = discRadius;
    double zoom = 300.0;//150.0;
    double camD = 20;
    //camPhi += 0.0006;

    static double revert_camPhi = false;
    if (!revert_camPhi) {
        camPhi += 0.0006;
        if (camTheta >= 10.0) {
            revert_camPhi = !revert_camPhi;
        }
    }
    else {
        camPhi -= 0.0006;
        if (camPhi <= 0.0) {
            revert_camPhi = !revert_camPhi;
        }
    }
    
    static double revert_camTheta = false;
    if (!revert_camTheta) {
        camTheta += 0.002;
        if (camTheta >= 0.155) {
            revert_camTheta = !revert_camTheta;
        }
    } else {
        camTheta -= 0.002;
        if (camTheta <= -0.155) {
            revert_camTheta = !revert_camTheta;
        }
    }

    std::vector<std::vector<sf::Vector3f>> xysIn;
    std::vector<std::vector<sf::Vector3f>> xysOut;
    int points = round(std::min(std::max(round(zoom / 2.0), (double)50.0), (double)400.0) * discQuality / (double)100.0);

    for (int order = 0; order < maxOrderDisc; order++) {
        std::vector<sf::Vector3f> xysIn1;
        std::vector<sf::Vector3f> xysOut1;
        for (int pointI = 0; pointI < points + 1; pointI++) {
            for (int inOut = 0; inOut < 2; inOut++) {

                double phi0 = 2.0 * PI * (double)pointI / (double)points - camPhi;

                double inclinaison = (double)sign(camTheta) * std::max((double)0.001, abs(camTheta));

                double phi;
                double s = sin(inclinaison);
                if (order == 0) {
                    phi = 2.0 * acot(s * cot(phi0 / 2.0));
                }
                else {
                    phi = acot(cot(phi0) / s);
                    if (sin(phi0) <= 0.0) {
                        phi += PI;
                    }
                }

                double r = rIn;
                if (inOut) {
                    r = rOut;
                }

                double x = r * cos(phi);
                double y = r * sin(phi);
                double z = 0.0;

                double x2 = x * cos(inclinaison) - z * sin(inclinaison);
                double z2 = z * cos(inclinaison) + x * sin(inclinaison);
                double y2 = y;

                double betaRot = atan2(z2, y2) + PI * (double)order;
                double alphaRot = atan2(sqrt(z2 * z2 + y2 * y2), x2);
                alphaRot *= pow(-1, order);
                alphaRot += PI * (double)(round(order + 1) - (double)((int)round(order + 1) % 2));

                double b = BfromRAlpha(r, alphaRot);

                double bmax = impactParameterApprox(PI / 2.0, camD);
                double rSeen = tan(asin(b / bmax)) * zoom;

                double xx = rSeen * cos(betaRot);
                double yy = rSeen * sin(betaRot);
                if (inOut) {
                    xysOut1.push_back(sf::Vector3f(xx, yy, phi + camPhi));
                }
                else {
                    xysIn1.push_back(sf::Vector3f(xx, yy, phi + camPhi));
                }
            }
        }
        xysOut.push_back(xysOut1);
        xysIn.push_back(xysIn1);
    }

    double sliceExpand = 0.2;
    double triangleExpand = 0.2;

    int discTexWidth = 1000;
    int discTexHeight = 1000;

    std::vector<std::vector<std::vector<sf::ConvexShape>>> shapes;

    for (int order = maxOrderDisc - 1; order >= 0; order--) {
        std::vector<std::vector<sf::ConvexShape>> shape1;
        for (int i = 0; i < points; i++) {
            std::vector<sf::ConvexShape> shape2;
            double startPhi = (double)((int)(xysIn[order][i].z / 2 / PI) % 1);
            while (startPhi < 0.0) { startPhi += 1.0; }
            double endPhi = std::min((double)1, startPhi + (double)1 / (double)points);
            double startU = startPhi * (double)discTexWidth;
            double endU = endPhi * (double)discTexWidth;

            std::vector<Vec4> pnts;

            double x = (double)window_size_X / 2.0 + (xysIn[order][i].x - sliceExpand * (xysIn[order][i + 1].x - xysIn[order][i].x));
            double y = (double)window_size_Y / 2.0 - (xysIn[order][i].y - sliceExpand * (xysIn[order][i + 1].y - xysIn[order][i].y));
            pnts.push_back(Vec4{ (double)x, (double)y, startU, 0 });

            x = (double)window_size_X / 2.0 + (xysOut[order][i].x - sliceExpand * (xysOut[order][i + 1].x - xysOut[order][i].x));
            y = (double)window_size_Y / 2.0 - (xysOut[order][i].y - sliceExpand * (xysOut[order][i + 1].y - xysOut[order][i].y));
            pnts.push_back(Vec4{ (double)x, (double)y, startU, (double)discTexHeight });

            x = (double)window_size_X / 2.0 + (xysOut[order][i + 1].x - sliceExpand * (xysOut[order][i].x - xysOut[order][i + 1].x));
            y = (double)window_size_Y / 2.0 - (xysOut[order][i + 1].y - sliceExpand * (xysOut[order][i].y - xysOut[order][i + 1].y));
            pnts.push_back(Vec4{ (double)x, (double)y, endU, (double)discTexHeight });

            x = (double)window_size_X / 2.0 + (xysIn[order][i + 1].x - sliceExpand * (xysIn[order][i].x - xysIn[order][i + 1].x));
            y = (double)window_size_Y / 2.0 - (xysIn[order][i + 1].y - sliceExpand * (xysIn[order][i].y - xysIn[order][i + 1].y));
            pnts.push_back(Vec4{ (double)x, (double)y, (double)endU, 0.0 });

            shape2 = textureMap(pnts, triangleExpand, 0.0, window);
            shape1.push_back(shape2);

        }
        shapes.push_back(shape1);
    }
    return shapes;
}

/*int __main()
{
    sf::ContextSettings settings;
    settings.antialiasingLevel = 16.0;
    sf::RenderWindow window(sf::VideoMode(window_size_X, window_size_Y), "Simulation trou noir", sf::Style::Close, settings);
    window.setFramerateLimit(30);


    sf::Image image;
    image.loadFromFile("include/texture.jpg");

    sf::Texture texture;
    texture.loadFromImage(image);

    sf::Sprite sprite;
    sprite.setTexture(texture);

    std::vector<std::vector<std::vector<sf::ConvexShape>>> shapes = computeDisk(window);

    for (int i = 0; i < shapes.size(); i++) {
        for (int j = 0; j < shapes[i].size(); j++) {
            for (int k = 0; k < shapes[i][j].size(); k++) {
                shapes[i][j][k].setTexture(&texture);
            }
        }
    }

    sf::Clock clock;
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        if (clock.getElapsedTime().asMilliseconds() >= 50) {
            //texture.loadFromImage(image);
            //sprite.setTexture(texture);
            window.clear();
            //window.draw(sprite);
            //drawDisk(window);

            for (int i = 0; i < shapes.size(); i++) {
                for (int j = 0; j < shapes[i].size(); j++) {
                    for (int k = 0; k < shapes[i][j].size(); k++) {
                        window.draw(shapes[i][j][k]);
                    }
                }
            }

            window.display();
            shapes = computeDisk(window);
            for (int i = 0; i < shapes.size(); i++) {
                for (int j = 0; j < shapes[i].size(); j++) {
                    for (int k = 0; k < shapes[i][j].size(); k++) {
                        shapes[i][j][k].setTexture(&texture);
                    }
                }
            }
            clock.restart();
        }
    }

    return 0;
}*/