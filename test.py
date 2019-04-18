import pyembroidery as em
import math

from collections import Counter
from itertools import *
import networkx as nx
import numpy as np

import glfw
import moderngl as gl
import imgui as gui
from imgui.integrations.glfw import GlfwRenderer
from pyrr import Matrix44 as mat4, matrix33 as mat3, Vector3 as vec3
import pywavefront as obj

from PIL import Image
# image = Image.open('teapot.png')
# image.show()

class RenderToTexture:
    def __init__(self, size=(256, 256), components=4, samples=4):
        self.ctx = ContextManager.get_default_context()
        self.texture = self.ctx.texture(size, components=4)
        self.fbo1 = self.ctx.simple_framebuffer(size, samples=samples)
        self.fbo2 = self.ctx.framebuffer(self.texture)
        self.scope = self.ctx.scope(self.fbo1)

    def __enter__(self):
        self.scope.__enter__()

    def __exit__(self, *args):
        self.scope.__exit__()
        self.ctx.copy_framebuffer(self.fbo2, self.fbo1)

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

resolution = (1000, 1000)
window = glfw.create_window(*resolution, "test", None, None)
glfw.make_context_current(window)

impl = GlfwRenderer(window)

ctx = gl.create_context()

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

clear_prog = ctx.program(
    vertex_shader='''
    #version 330

    in vec2 V2F;

    void main()
    {
        gl_Position = vec4(V2F.xy, 0, 1);
    }
   ''',
    fragment_shader='''
    #version 330

    out int face_id;

    void main() {
        face_id = -1;
    }
    ''',
)

face_prog = ctx.program(
    vertex_shader='''
    #version 330
    uniform mat4 MVP;

    in vec3 V3F;

    void main() {
        gl_Position = MVP * vec4(V3F, 1.0);
    }
    ''',
    fragment_shader='''
    #version 330

    out int face_id;

    void main() {
        face_id = gl_PrimitiveID;
    }
    ''',
)

# load obj file
# display object with camera controls and basic normals shading
# button to toggle wireframe view
# have a capture button
# convert to wireframe (quads?) and do backface culling on cpu to get edges
# perform partial graphics pipeline on cpu to render to a 1000x1000 grid
# perform a walk of the edges

scene = obj.Wavefront('teapot.obj', collect_faces=True)
# scene = obj.Wavefront('teapot_simple.obj', collect_faces=True)
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

# face_fbo = ctx.framebuffer(ctx.renderbuffer((1000, 1000)))
# face_fbo = ctx.simple_framebuffer((1000,1000), components=1, samples=0, dtype='f4')
face_fbo = ctx.simple_framebuffer((1000,1000), components=1, samples=0, dtype='i4')

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
face_vao = ctx.simple_vertex_array(face_prog, vbo, 'V3F', index_buffer=ibo)

fs_tri = np.array([
    -1.0, -1.0,
     3.0, -1.0,
    -1.0,  3.0,
])
clear_vao = ctx.simple_vertex_array(clear_prog, ctx.buffer(fs_tri.astype('f4').tobytes()), 'V2F')

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
    global view, proj, resolution

    clip_pos = proj.T.dot(view.T.dot(np.hstack((world_pos, [1.0]))))
    ndc_pos = np.array([ c/clip_pos[3] for c in clip_pos ])
    screen_pos = np.array([ (ndc_pos[i] + 1.0) / 2.0 * resolution[i] for i in range(2) ])

    return screen_pos.astype(int)


def create_pes():
    global mvp
    pattern = em.EmbPattern()

    vertices = [ world_space_to_screen_space(vert) for vert in scene.vertices ]

    def to_i4(l):
        return l[0] | l[1] << 8 | l[2] << 16 | l[3] << 24

    im = list(face_fbo.read(components=1, dtype='i4'))
    visible_faces = Counter([ to_i4(im[k:k+4]) for k in range(0,len(im),4) ])
    del visible_faces[0xFFFFFFFF]
    # TODO count pixels of visible faces and prune any with fewer than some number
    print(visible_faces)

    faces = visible_faces
    # threshold = 10
    # faces = [ f for f,cnt in visible_faces.items() if cnt >= threshold ]

    stitches = []
    G = nx.Graph()

    # go through all things in scene indices
    for idx in faces:
        face = scene.mesh_list[0].faces[idx]

        verts = [ tuple(vertices[p]) for p in face ]
        edges = list(zip(verts, islice(cycle(verts), 1, None)))

        G.add_edges_from(edges)

        for (p0, p1) in edges:
            # needs to be at least 2
            for t in np.linspace(0, 1, 2):
                stitches.append(((1-t)*p0[0] + t*p1[0], (1-t)*p0[1] + t*p1[1]))

    stitches2 = []
    for C in [ G.subgraph(c) for c in nx.connected_components(G) ]:
        for (p0, p1) in nx.eulerian_circuit(nx.eulerize(C)):
            # needs to be at least 2
            for t in np.linspace(0, 1, 2):
                stitches2.append(((1-t)*p0[0] + t*p1[0], (1-t)*p0[1] + t*p1[1]))
    stitches = [ k for k,g in groupby(stitches) ]
    stitches2 = [ k for k,g in groupby(stitches2) ]
    print(len(stitches), len(stitches2))

    # del stitches[len(stitches)-1] # if you do it it generates more like a quadmesh
    for (x, y) in stitches2:
        pattern.add_stitch_absolute(em.STITCH, x, y)

    # TODO prettification:
    # backface culling
    # only draw one face per pixel? just remove all hidden faces including the backfaces and areas that are dense on faces, so check face ids on the final view and then only draw those faces that are present.

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
    face_prog['MVP'].write(mvp.astype('f4').tobytes())

    # ctx.enable_only(gl.NOTHING)
    ctx.enable_only(gl.DEPTH_TEST | gl.CULL_FACE)

    # draw GUI
    ctx.screen.use()
    gui.new_frame()
    gui.begin("Your first window!")
    gui.text("Hello world!")
    gui.end()

    ctx.clear()
    ctx.wireframe = True
    vao.render()
    ctx.wireframe = False

    face_fbo.clear(0)
    face_fbo.use()
    ctx.disable(gl.DEPTH_TEST)
    clear_vao.render()
    ctx.enable_only(gl.DEPTH_TEST | gl.CULL_FACE)
    face_vao.render()

    ctx.screen.use()

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
