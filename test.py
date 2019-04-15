import pyembroidery as em

import networkx as nx
import numpy as np

import glfw
import moderngl as gl
import imgui as gui
from imgui.integrations.glfw import GlfwRenderer
from pyrr import Matrix44 as mat4, Vector3 as vec3
import pywavefront as obj

# from PIL import Image
# image = Image.open('teapot.png')
# image.show()

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

resolution = (640, 480)
window = glfw.create_window(*resolution, "test", None, None)
glfw.make_context_current(window)

impl = GlfwRenderer(window)

ctx = gl.create_context()
# ctx.enable_only(gl.DEPTH_TEST | gl.CULL_FACE)

prog = ctx.program(
    vertex_shader='''
    #version 330
    uniform mat4 MVP;

    in vec3 v_pos;
    in vec3 v_col;

    out vec3 v_color;    // Goes to the fragment shader

    void main() {
        gl_Position = MVP * vec4(v_pos, 1.0);
        v_color = v_col;
    }
    ''',
    fragment_shader='''
    #version 330
    in vec3 v_color;

    out vec4 f_color;

    void main() {
        f_color = vec4(v_color, 1.0);
    }
    ''',
)

# Point coordinates are put followed by the vec3 color values
vertices = np.array([
    # x, y, red, green, blue
    0.0, 0.8, 1.0, 0.0, 0.0,
    -0.6, -0.8, 0.0, 1.0, 0.0,
    0.6, -0.8, 0.0, 0.0, 1.0,
])

indices = np.array([
    0, 1, 2,
])

vertices = []
indices = []

# load obj file
# display object with camera controls and basic shading
# button to toggle wireframe view
# have a capture button
# convert to wireframe (quads?) and do backface culling on cpu to get edges
# perform partial graphics pipeline on cpu to render to a 1000x1000 grid
# perform a walk of the edges
min_pos = np.array([0.0, 0.0, 0.0])
max_pos = np.array([0.0, 0.0, 0.0])

model = obj.Wavefront('models/teapot.obj')

for name, material in model.materials.items():
    pass

center = (min_pos + max_pos)/2

# TODO make a zoom to fit function, take the bounding box and figure
# out distance for the entire object to be within the frustum of the
# camera

vertices = np.array(vertices)
indices = np.array(indices)

# might be able to use VertexArray.transform to get the vertices in
# transformed coordinates after capture?

vbo = ctx.buffer(vertices.astype('f4').tobytes())
ibo = ctx.buffer(indices.astype('uint32').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'v_pos', 'v_col', index_buffer=ibo)

model = np.identity(4)
view = mat4.look_at(vec3([4.0, 5.0, 2.0]), vec3(center), vec3([0.0, 1.0, 0.0]))
proj = mat4.perspective_projection(50, resolution[0]/resolution[1], 0.1, 100.0)
mvp = proj * view * model
print("M"),   print(model)
print("V"),   print(view)
print("P"),   print(proj)
print("MVP"), print(mvp)

# prog['MVP'].value = tuple(mvp.flatten())
prog['MVP'].write(mvp.astype('f4').tobytes())

gui.create_context()

while not glfw.window_should_close(window):
    glfw.poll_events()
    impl.process_inputs()

    gui.new_frame()
    gui.begin("Your first window!")
    gui.text("Hello world!")
    gui.end()

    ctx.clear(0.0, 0.0, 0.0)
    vao.render()
    # vao.render(gl.LINE_STRIP)

    gui.render()
    impl.render(gui.get_draw_data())

    glfw.swap_buffers(window)

glfw.terminate()

G = nx.Graph()

# old version
# G = nx.eulerize(G)

# add each edge to the graph twice to guarantee Eulerian and even
# weight on all edges
G = nx.MultiGraph()

for u,v in nx.eulerian_circuit(G):
    pass

for u,v in nx.eulerian_circuit(G):
    pass

## future ideas

# depth based width, might need 3 functions

# can double up the edges so that you are guaranteed an eulerian path
# and have equal weight on all edges

# can make nice embroidery by making the dual graph of the multigraph,
# where the nodes are the two edges between nodes in the original
# graph, and the edges between those nodes are the "nice" nodes in
# terms of appearance for embroidery. so you can go back on the same
# edge, you can take edges on the same side of a vertex, but you can't
# take an edge that passes another edge.

# steps to take:
# - double all the edges in the original graph
# - convert to a line graph of that graph (the edge-vertex dual)
# - look for hamiltonian paths in the line graph
