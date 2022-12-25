import numpy as np
import cv2 as cv
import svg.path
import xml.etree.ElementTree as ET
import cmath
from time import perf_counter
import sys

offset = -(50+50j)
width, height = 800, 800
t_scale = 0.001
scale = 3
n = 50
res = 10
file = 'pi.svg'

for i, arg in enumerate(sys.argv[1::2]):
    i = 2*i + 1
    match (arg):
        case '-f':
            file = sys.argv[i+1]
        case '--Scale' | '-s':
            scale = float(sys.argv[i+1])
        case '--Resolution' | '-r':
            res = float(sys.argv[i+1])
        case '--Terms' | '-t':
            n = int(sys.argv[i+1])
        case '--Time' | '-ts':
            t_scale = float(sys.argv[i+1])
        case '--Offset' | '-o':
            num = sys.argv[i+1]
            offset = complex(float(num.split('+')[0]),
                             float(num.split('+')[1]))


cubic_bezier_matrix = np.array([
    [-1,  3, -3,  1],
    [3, -6,  3,  0],
    [-3,  3,  0,  0],
    [1,  0,  0,  0]
])


def cubic_bezier_sample(start, control1, control2, end):
    inputs = np.array([start, control1, control2, end])
    partial = cubic_bezier_matrix.dot(inputs)

    return (lambda t: np.array([t**3, t**2, t, 1]).dot(partial))


def quadratic_sample(start, control, end):
    # Quadratic bezier curve is just cubic bezier curve
    # with the same control points.
    return cubic_bezier_sample(start, control, control, end)


# Parse the SVG file
tree = ET.parse(file)
root = tree.getroot()

img = np.zeros((width, height, 3), dtype=np.uint8)
path_data = []
# Iterate through all the 'path' elements
for child in root:
    if 'path' in child.tag:
        global points
        path = child
        # Get the 'd' attribute, which contains the path data
        d = path.get('d')

        path_data = [p for p in svg.path.parse_path(d)]
        # path_data.
        # points = np.array([p.end-offset for p in svg.path.parse_path(d)])
    if 'g' in child.tag:
        # Import the necessary modules
        g_elements = child
        # Get the points from the 'g' element
        # Loop through the 'g' elements
        for g_element in g_elements:
            # Get the points from the 'g' element
            if 'path' in g_element.tag:
                path_data += [
                    p for p in svg.path.parse_path(g_element.get('d'))]
            for path in g_element:
                if 'path' in path.tag:
                    path_data += [
                        p for p in svg.path.parse_path(path.get('d'))]

print(len(path_data))
points = []

for p in path_data:
    if type(p) == svg.path.path.CubicBezier:
        curve = cubic_bezier_sample(p.start, p.control1, p.control2, p.end)
        points += [curve(t) for t in np.linspace(0, 1, int(res))]
    if type(p) == svg.path.path.QuadraticBezier:
        curve = quadratic_sample(p.start, p.control, p.end)
        points += [curve(t) for t in np.linspace(0, 1, int(res))]

points = np.array(points,dtype=np.csingle)
print(len(points))
# Compute the coefficients of the fourier series
c = np.zeros((2*n+1), dtype=complex)
for i in range(-n, n+1):
    print(f'\r{n+i}/{2*n}',end='')
    c[n+i] = np.sum(np.fromiter((p*cmath.exp(-i*2*cmath.pi*1j*j/len(points)) /
                    len(points) for j, p in enumerate(points)), dtype=np.csingle))
points_t = np.copy(c)

WHITE = (255, 255, 255)

t = 0
dt = 0
output = np.copy(img)
p1, p2 = (0, 0), (0, 0)
t0 = perf_counter()
while 1:
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    cv.imshow('Fourier Series', img)
    img *= 0

    # Draw first vector
    vector_start = scale*(points_t[n]+offset)
    # cv.line(img, (width//2, height//2), (int(width//2+vector_start.real),
    #         int(height//2+vector_start.imag)), WHITE, 1)

    # Draw remaining vectors
    for i in range(1, n+1):
        vector_end = points_t[n+i] + points_t[n-i]
        cv.line(img, (int(width/2+vector_start.real), int(height/2+vector_start.imag)),
                (int(width/2+vector_start.real+scale*vector_end.real),
                int(height/2+vector_start.imag+scale*vector_end.imag)),
                WHITE, 1)
        vector_start += vector_end*scale

    # Rotate all vectors
    for i in range(-n, n):
        points_t[n+i] *= cmath.exp(i*2*cmath.pi*1j*t)

    # Draw output image
    p1 = p2
    p2 = vector_start
    if t > dt*t_scale:
        cv.line(output, (int(p1.real+width/2), int(p1.imag+height/2)),
                (int(p2.real+width/2), int(p2.imag+height/2)), WHITE, 1)
        # output[int(p2.imag+width/2), int(p2.real+height/2)] = WHITE
        cv.add(img, output, img)

    dt = perf_counter()-t0
    t0 = perf_counter()
    t += t_scale*dt
