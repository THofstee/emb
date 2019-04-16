import pyembroidery as em
import math

import networkx as nx
import numpy as np

import glfw
import moderngl as gl
import imgui as gui
from imgui.integrations.glfw import GlfwRenderer
from pyrr import Matrix44 as mat4, matrix33 as mat3, Vector3 as vec3
import pywavefront as obj

# from PIL import Image
# image = Image.open('teapot.png')
# image.show()

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

resolution = (1000, 1000)
window = glfw.create_window(*resolution, "test", None, None)
glfw.make_context_current(window)

impl = GlfwRenderer(window)

ctx = gl.create_context()
# ctx.enable_only(gl.DEPTH_TEST | gl.CULL_FACE)

prog = ctx.program(
    vertex_shader='''
    #version 330
    uniform mat4 MVP;

    in vec2 T2F;
    in vec3 N3F;
    in vec3 V3F;

    out vec2 v_tangent;
    out vec3 v_normal;
    out vec3 v_color;    // Goes to the fragment shader

    void main() {
        v_tangent = T2F;
        v_normal = N3F;
        gl_Position = MVP * vec4(V3F, 1.0);
        v_color = vec3(0.4, 0.4, 0.4);
    }
    ''',
    fragment_shader='''
    #version 330
    in vec2 v_tangent;
    in vec3 v_normal;
    in vec3 v_color;

    out vec4 f_color;

    void main() {
        f_color = vec4(v_color, 1.0);
    }
    ''',
)

# load obj file
# display object with camera controls and basic shading
# button to toggle wireframe view
# have a capture button
# convert to wireframe (quads?) and do backface culling on cpu to get edges
# perform partial graphics pipeline on cpu to render to a 1000x1000 grid
# perform a walk of the edges

scene = obj.Wavefront('teapot.obj', collect_faces=True)
# scene = obj.Wavefront('models/cube.obj', collect_faces=True)

# scene.vertices
# scene.meshes
# scene.mesh_list
# scene.materials
# scene.mtllibs

min_pos = np.array([0.0, 0.0, 0.0])
max_pos = np.array([0.0, 0.0, 0.0])
for vert in scene.vertices:
    min_pos = np.minimum(min_pos, vert)
    max_pos = np.maximum(max_pos, vert)

center = (min_pos + max_pos)/2

vertices = np.array([ v for vert in scene.vertices for v in vert ])
indices = np.array([ f for face in scene.mesh_list[0].faces for f in face ])

# for name, material in scene.materials.items():
#     print(material.__dict__)
#     print(prog._members)
#     vertices = np.array(material.vertices)
#     vbo = ctx.buffer(vertices.astype('f4').tobytes())
#     vao = ctx.simple_vertex_array(prog, vbo, 'T2F', 'N3F', 'V3F')
#     break

# TODO make a zoom to fit function, take the bounding box and figure
# out distance for the entire object to be within the frustum of the
# camera, just do something like 80% fov_y and 80% fov_x then pick the
# required distance. Might need to make camera aligned bounding box
# though.

# # might be able to use VertexArray.transform to get the vertices in
# # transformed coordinates after capture?

vbo = ctx.buffer(vertices.astype('f4').tobytes())
ibo = ctx.buffer(indices.astype('uint32').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'V3F', index_buffer=ibo)

print(min_pos, max_pos, center)

model = np.identity(4)
eye = np.copy(center)
eye[2] -= 3*min_pos[2]
up = [0.0, 1.0, 0.0]
print(center, eye)
# eye = [ 0.0, 0.0, 120.0 ]
view = mat4.look_at(vec3(eye), vec3(center), vec3(up))
proj = mat4.perspective_projection(50, resolution[0]/resolution[1], 0.01, 1000.0)
mvp = proj * view * model
print("M"),   print(model)
print("V"),   print(view)
print("P"),   print(proj)
print("MVP"), print(mvp)

# prog['MVP'].value = tuple(mvp.flatten())

gui.create_context()

def key_event(window, key, scancode, action, mode):
    global eye

    if action == glfw.PRESS and key == glfw.KEY_RIGHT:
        rot = 5
        rot = mat3.create_from_axis_rotation(vec3([0.0, 1.0, 0.0]), np.radians(rot))
        eye = rot.dot(eye)
    elif action == glfw.PRESS and key == glfw.KEY_LEFT:
        rot = -5
        rot = mat3.create_from_axis_rotation(vec3([0.0, 1.0, 0.0]), np.radians(rot))
        eye = rot.dot(eye)

    if action == glfw.PRESS and key == glfw.KEY_UP:
        rot = 5
        right = np.cross(center - eye, up)
        rot = mat3.create_from_axis_rotation(vec3(right), np.radians(rot))
        eye = rot.dot(eye)
    elif action == glfw.PRESS and key == glfw.KEY_DOWN:
        rot = -5
        right = np.cross(center - eye, up)
        rot = mat3.create_from_axis_rotation(vec3(right), np.radians(rot))
        eye = rot.dot(eye)

    if action == glfw.PRESS and key == glfw.KEY_S:
        eye = 1.01 * eye
    elif action == glfw.PRESS and key == glfw.KEY_W:
        eye = 0.99 * eye

    if action == glfw.PRESS and key == glfw.KEY_SPACE:
        create_pes()

def world_space_to_screen_space(world_pos):
    global proj, resolution, eye, center
    view = mat4.look_at(vec3(eye), vec3(center), vec3([0.0, 1.0, 0.0]))

    print("w", world_pos)

    clip_pos = proj.T.dot(view.T.dot(np.hstack((world_pos, [1.0]))))
    print("c", clip_pos)

    ndc_pos = np.array([ c/clip_pos[3] for c in clip_pos ])
    print("n", ndc_pos)

    screen_pos = np.array([ (ndc_pos[i] + 1.0) / 2.0 * resolution[i] for i in range(2) ])
    print("s", screen_pos)

    return screen_pos.astype(int)


def create_pes():
    global mvp
    pattern = em.EmbPattern()

    # # multiply vertices by mvp
    # vertices = [ np.array(mvp.dot(np.hstack((vert, np.array([1.0]))))) for vert in scene.vertices ]

    # print("verts")
    # print(vertices)
    # clip_verts = []
    # for vert in vertices:
    #     clip_verts += [np.array([vert[0]/vert[3], vert[1]/vert[3], vert[2]/vert[3], vert[3]/vert[3]])]

    # # vertices = clip_verts

    # min_pos = np.array([0.0, 0.0, 0.0, 0.0])
    # max_pos = np.array([0.0, 0.0, 0.0, 0.0])
    # for vert in vertices:
    #     min_pos = np.minimum(min_pos, vert)
    #     max_pos = np.maximum(max_pos, vert)

    # # BEGIN HACK
    # vertices = [ np.array([vert[0], vert[1]]) for vert in vertices ]

    # min_pos = np.array([0.0, 0.0])
    # for vert in vertices:
    #     min_pos = np.minimum(min_pos, vert)

    # for vert in vertices:
    #     vert -= min_pos

    # max_pos = np.array([0.0, 0.0])
    # for vert in vertices:
    #     max_pos = np.maximum(max_pos, vert)

    # for vert in vertices:
    #     vert /= max(max_pos)
    # # END HACK

    # maybe this works better?
    vertices = [ world_space_to_screen_space(vert) for vert in scene.vertices ]
    print(vertices)

    # go through all things in scene indices
    def to_abs(x, y):
        return x, y
        # return int(math.floor(x * 1000)), int(math.floor(y * 1000))

    for face in scene.mesh_list[0].faces:
        # take x/y and round onto 1000x1000 grid
        pattern.add_stitch_absolute(em.STITCH, *to_abs(vertices[face[0]][0], vertices[face[0]][1]))
        # print("STITCH", *to_abs(vertices[face[0]][0], vertices[face[0]][1]))
        pattern.add_stitch_absolute(em.STITCH, *to_abs(vertices[face[1]][0], vertices[face[1]][1]))
        # print("STITCH", *to_abs(vertices[face[1]][0], vertices[face[1]][1]))
        pattern.add_stitch_absolute(em.STITCH, *to_abs(vertices[face[2]][0], vertices[face[2]][1]))
        # print("STITCH", *to_abs(vertices[face[2]][0], vertices[face[2]][1]))
        pattern.add_stitch_absolute(em.STITCH, *to_abs(vertices[face[0]][0], vertices[face[0]][1]))
        # print("STITCH", *to_abs(vertices[face[0]][0], vertices[face[0]][1]))

    # create dst format (pes doesn't work?)
    print("Saving file...")
    em.write_dst(pattern, 'file.dst')
    em.write_pes(pattern, 'file.pes')
    em.write_txt(pattern, 'file.txt')

    print("Done.")

# glfw.set_input_mode(window, glfw.STICKY_KEYS, True)gglfw
glfw.set_key_callback(window, key_event)

while not glfw.window_should_close(window):
    glfw.poll_events()
    impl.process_inputs()

    # recompute MVP
    view = mat4.look_at(vec3(eye), vec3(center), vec3([0.0, 1.0, 0.0]))
    mvp = proj * view * model
    prog['MVP'].write(mvp.astype('f4').tobytes())

    # draw GUI
    gui.new_frame()
    gui.begin("Your first window!")
    gui.text("Hello world!")
    gui.end()

    ctx.clear(0.0, 0.0, 0.0)
    # vao.render()
    vao.render(gl.LINE_STRIP)

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
