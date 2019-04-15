import numpy as np

def vec3(x, y, z):
    return np.array([x, y, z])

def look_at(eye, center = vec3(0, 0, 0), up = vec3(0, 1, 0)):
    camera_direction = eye - center
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    camera_right = np.cross(up, camera_direction)
    camera_up = np.cross(camera_direction, camera_right)

    m1 = np.array([
        [ camera_right[0], camera_right[1], camera_right[2], 0 ],
        [ camera_up[0], camera_up[1], camera_up[2], 0 ],
        [ camera_direction[0], camera_direction[1], camera_direction[2], 0 ],
        [ 0, 0, 0, 1 ]
    ])

    print(m1)

    m2 = np.array([
        [ 1, 0, 0, -eye[0] ],
        [ 0, 1, 0, -eye[1] ],
        [ 0, 0, 1, -eye[2] ],
        [ 0, 0, 0, 1 ]
    ])

    print(m2)

    return m1.dot(m2)

def perspective(fov_y, aspect, near, far):
    return np.array([
        [ 1 / (aspect * np.tan(fov_y/2)), 0, 0, 0 ],
        [ 0, 1 / np.tan(fov_y/2), 0, 0 ],
        [ 0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near) ],
        [ 0, 0, -1, 0 ],
    ])
