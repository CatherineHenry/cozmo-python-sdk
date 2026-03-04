# Copyright (c) 2017 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''This module provides a 3D visualizer for Cozmo's world state.

It uses PyOpenGL, a Python OpenGL 3D graphics library which is available on most
platforms. It also depends on the Pillow library for image processing.

The easiest way to make use of this viewer is to call :func:`cozmo.run_program`
with `use_3d_viewer=True` or :func:`cozmo.run.connect_with_3dviewer`.

Warning:
    This package requires Python to have the PyOpenGL package installed, along
    with an implementation of GLUT (OpenGL Utility Toolkit).

    To install the Python packages do ``pip3 install --user "cozmo[3dviewer]"``

    On Windows and Linux you must also install freeglut (macOS / OSX has one
    preinstalled).

    On Linux: ``sudo apt-get install freeglut3``

    On Windows: Go to http://freeglut.sourceforge.net/ to get a ``freeglut.dll``
    file. It's included in any of the `Windows binaries` downloads. Place the DLL
    next to your Python script, or install it somewhere in your PATH to allow any
    script to use it."
'''


# __all__ should order by constants, event classes, other classes, functions.
__all__ = ['DynamicTexture', 'LoadedObjFile', 'OpenGLViewer', 'OpenGLWindow',
           'RenderableObject',
           'LoadMtlFile']


import collections
import math
import random
from math import cos, sin, pi
import time
from threading import Event

from pkg_resources import resource_stream

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PIL import Image

from .exceptions import InvalidOpenGLGlutImplementation, RobotBusy
from . import logger
from . import nav_memory_map
from . import objects
from . import robot
from . import util
from . import world
from .robot import LiftPosition

import matplotlib.pyplot as plt
from matplotlib import cm, colormaps

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)
import sys
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtOpenGL import QOpenGLWindow
from OpenGL.GL import glClear, glClearColor, GL_COLOR_BUFFER_BIT



# Check if OpenGL imported correctly and bound to a valid GLUT implementation


def _glut_install_instructions():
    if sys.platform.startswith('linux'):
        return "Install freeglut: `sudo apt-get install freeglut3`"
    elif sys.platform.startswith('darwin'):
        return "GLUT should already be installed by default on macOS!"
    elif sys.platform in ('win32', 'cygwin'):
        return "Install freeglut: You can download it from http://freeglut.sourceforge.net/ \n" \
               "You just need the `freeglut.dll` file, from any of the 'Windows binaries' downloads. " \
               "Place the DLL next to your Python script, or install it somewhere in your PATH " \
               "to allow any script to use it."
    else:
        return "(Instructions unknown for platform %s)" % sys.platform


def _verify_glut_init():
    # According to the documentation, just checking bool(glutInit) is supposed to be enough
    # However on Windows with no GLUT DLL that can still pass, even if calling the method throws a null function error.
    if bool(glutInit):
        try:
            glutInit()
            return True
        except OpenGL.error.NullFunctionError as e:
            pass

    return False


if not _verify_glut_init():
    raise InvalidOpenGLGlutImplementation(_glut_install_instructions())


_resource_package = __name__  # All resources are in subdirectories from this file's location

class DynamicTexture:
    """Wrapper around An OpenGL Texture that can be dynamically updated."""

    def __init__(self):
        self._texId =  glGenTextures(1)
        self._width = None
        self._height = None
        # Bind an ID for this texture
        glBindTexture(GL_TEXTURE_2D, self._texId)
        # Use bilinear filtering if the texture has to be scaled
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def bind(self):
        """Bind the texture for rendering."""
        glBindTexture(GL_TEXTURE_2D, self._texId)

    def update(self, pil_image: Image.Image):
        """Update the texture to contain the provided image.

        Args:
            pil_image (PIL.Image.Image): The image to write into the texture.
        """
        # Ensure the image is in RGBA format and convert to the raw RGBA bytes.
        image_width, image_height = pil_image.size
        image = pil_image.convert("RGBA").tobytes("raw", "RGBA")

        # Bind the texture so that it can be modified.
        self.bind()
        if (self._width==image_width) and (self._height==image_height):
            # Same size - just need to update the texels.
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height,
                            GL_RGBA, GL_UNSIGNED_BYTE, image)
        else:
            # Different size than the last frame (e.g. the Window is resizing)
            # Create a new texture of the correct size.
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, image)

        self._width = image_width
        self._height = image_height


def LoadMtlFile(filename):
    """Load a .mtl material file, and return the contents as a dictionary.

    Supports the subset of MTL required for the Cozmo 3D viewer assets.

    Args:
        filename (str): The filename of the file to load.

    Returns:
        dict: A dictionary mapping named MTL attributes to values.
    """
    contents = {}
    current_mtl = None

    resource_path = '/'.join(('assets', filename))  # Note: Deliberately not os.path.join, for use with pkg_resources
    file_data = resource_stream(_resource_package, resource_path)

    for line in file_data:
        line = line.decode("utf-8")  # Convert bytes line to a string
        if line.startswith('#'):
            # ignore comments in the file
            continue
        values = line.split()
        if not values:
            # ignore empty lines
            continue
        attribute_name = values[0]
        if attribute_name == 'newmtl':
            # Create a new empty material
            current_mtl = contents[values[1]] = {}
        elif current_mtl is None:
            raise ValueError("mtl file must start with newmtl statement")
        elif attribute_name == 'map_Kd':
            # Diffuse texture map - load the image into memory
            image_name = values[1]
            image_resource_path = '/'.join(('assets', image_name))  # Note: Deliberately not os.path.join, for use with pkg_resources
            image_file_data = resource_stream(_resource_package, image_resource_path)
            with Image.open(image_file_data) as image:
                image_width, image_height = image.size
                image = image.convert("RGBA").tobytes("raw", "RGBA")

            # Bind the image as a texture that can be used for rendering
            texture_id =  glGenTextures(1)
            current_mtl['texture_Kd'] = texture_id

            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        else:
            # Store the values for this attribute as a list of float values
            current_mtl[attribute_name] = list(map(float, values[1:]))
    # File loaded successfully - return the contents
    return contents


class LoadedObjFile:
    """The loaded / parsed contents of a 3D Wavefront OBJ file.

    This is the intermediary step between the file on the disk, and a renderable
    3D object. It supports the subset of the OBJ file that was used in the
    Cozmo and Cube assets, and does not attempt to exhaustively support every
    possible setting.

    Args:
        filename (str): The filename of the OBJ file to load.
    """
    def __init__(self, filename):
        # list: The vertices (each vertex stored as list of 3 floats).
        self.vertices = []
        # list: The vertex normals (each normal stored as list of 3 floats).
        self.normals = []
        # list: The texture coordinates (each coordinate stored as list of 2 floats).
        self.tex_coords = []
        # dict: The faces for each mesh, indexed by mesh name.
        self.mesh_faces = {}

        # dict: A dictionary mapping named MTL attributes to values.
        self.mtl = None

        group_name = None
        material = None

        resource_path = '/'.join(('assets', filename))  # Note: Deliberately not os.path.join, for use with pkg_resources
        file_data = resource_stream(_resource_package, resource_path)

        for line in file_data:
            line = line.decode("utf-8")  # Convert bytes to string
            if line.startswith('#'):
                # ignore comments in the file
                continue

            values = line.split()
            if not values:
                # ignore empty lines
                continue

            if values[0] == 'v':
                # vertex position
                v = list(map(float, values[1:4]))
                self.vertices.append(v)
            elif values[0] == 'vn':
                # vertex normal
                v = list(map(float, values[1:4]))
                self.normals.append(v)
            elif values[0] == 'vt':
                # texture coordinate
                self.tex_coords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                # material
                material = values[1]
            elif values[0] == 'mtllib':
                # material library (a filename)
                self.mtl = LoadMtlFile(values[1])
            elif values[0] == 'f':
                # A face made up of 3 or 4 vertices - e.g. `f v1 v2 v3` or `f v1 v2 v3 v4`
                # where each vertex definition is multiple indexes seperated by
                # slashes and can follow the following formats:
                # position_index
                # position_index/tex_coord_index
                # position_index/tex_coord_index/normal_index
                # position_index//normal_index

                positions = []
                tex_coords = []
                normals = []

                for vertex in values[1:]:
                    vertex_components = vertex.split('/')

                    positions.append(int(vertex_components[0]))

                    # There's only a texture coordinate if there's at least 2 entries and the 2nd entry is non-zero length
                    if len(vertex_components) >= 2 and len(vertex_components[1]) > 0:
                        tex_coords.append(int(vertex_components[1]))
                    else:
                        # OBJ file indexing starts at 1, so use 0 to indicate no entry
                        tex_coords.append(0)

                    # There's only a normal if there's at least 2 entries and the 2nd entry is non-zero length
                    if len(vertex_components) >= 3 and len(vertex_components[2]) > 0:
                        normals.append(int(vertex_components[2]))
                    else:
                        # OBJ file indexing starts at 1, so use 0 to indicate no entry
                        normals.append(0)

                try:
                    mesh_face = self.mesh_faces[group_name]
                except KeyError:
                    # Create a new mesh group
                    self.mesh_faces[group_name] = []
                    mesh_face = self.mesh_faces[group_name]

                mesh_face.append((positions, normals, tex_coords, material))
            elif values[0] == 'o':
                # object name - ignore
                pass
            elif values[0] == 'g':
                # group name (for a sub-mesh)
                group_name = values[1]
            elif values[0] == 's':
                # smooth shading (1..20, and 'off') - ignore
                pass
            else:
                logger.warning("LoadedObjFile Ignoring unhandled type '%s' in line %s",
                               values[0], values)


class RenderableObject:
    """Container for an object that can be rendered via OpenGL.

    Can contain multiple meshes, for e.g. articulated objects.

    Args:
        object_data (LoadedObjFile): The object data (vertices, faces, etc.)
            to generate the renderable object from.
        override_mtl (dict): An optional material to use as an override instead
            of the material specified in the data. This allows one OBJ file
            to be used to create multiple objects with different materials
            and textures. Use :meth:`LoadMtlFile` to generate a dict from a
            MTL file.
    """
    def __init__(self, object_data: LoadedObjFile, override_mtl=None):
        #: dict: The individual meshes, indexed by name, for this object.
        self.meshes = {}
        mtl_dict = override_mtl if (override_mtl is not None) else object_data.mtl

        def _as_rgba(color):
            if len(color) >= 4:
                return color
            else:
                # RGB - add alpha defaulted to 1
                return color + [1.0]

        for key in object_data.mesh_faces:
            new_gl_list = glGenLists(1)
            glNewList(new_gl_list, GL_COMPILE)

            self.meshes[key] = new_gl_list

            part_faces = object_data.mesh_faces[key]

            glEnable(GL_TEXTURE_2D)
            glFrontFace(GL_CCW)

            for face in part_faces:
                vertices, normals, texture_coords, material = face

                mtl = mtl_dict[material]
                if 'texture_Kd' in mtl:
                    # use diffuse texture map
                    glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
                else:
                    # No texture map
                    glBindTexture(GL_TEXTURE_2D, 0)

                # Diffuse light
                mtl_kd_rgba = _as_rgba(mtl['Kd'])
                glColor(mtl_kd_rgba)

                # Ambient light
                if 'Ka' in mtl:
                    mtl_ka_rgba = _as_rgba(mtl['Ka'])
                    glMaterialfv(GL_FRONT, GL_AMBIENT, mtl_ka_rgba)
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, mtl_kd_rgba)
                else:
                    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mtl_kd_rgba);

                # Specular light
                if 'Ks' in mtl:
                    mtl_ks_rgba = _as_rgba(mtl['Ks'])
                    glMaterialfv(GL_FRONT, GL_SPECULAR, mtl_ks_rgba);
                    if 'Ns' in mtl:
                        specular_exponent = mtl['Ns']
                        glMaterialfv(GL_FRONT, GL_SHININESS, specular_exponent);

                # Polygon (N verts) with optional normals and tex coords
                glBegin(GL_POLYGON)
                for i in range(len(vertices)):
                    normal_index = normals[i]
                    if normal_index > 0:
                        glNormal3fv(object_data.normals[normal_index - 1])
                    tex_coord_index = texture_coords[i]
                    if tex_coord_index > 0:
                        glTexCoord2fv( object_data.tex_coords[tex_coord_index - 1])
                    glVertex3fv(object_data.vertices[vertices[i] - 1])
                glEnd()

            glDisable(GL_TEXTURE_2D)
            glEndList()

    def draw_all(self):
        """Draw all of the meshes."""
        for mesh in self.meshes.values():
            glCallList(mesh)


def _make_origin_arrow():
    new_gl_list = glGenLists(1)
    glNewList(new_gl_list, GL_COMPILE)

    glBegin(GL_TRIANGLES)

    # below grid
    glVertex3f(0, 10, -0.5)
    glVertex3f(0, -10, -0.5)
    glVertex3f(10, 0, -0.5)

    #above grid
    glVertex3f(0, 10, 0.0)
    glVertex3f(0, -10, 0.0)
    glVertex3f(10, 0, 0.0)

    glEnd()

    glEndList()

    return new_gl_list


def _make_pose_arrow():
    """Make a small-size pose cube, with normals, centered at the origin"""
    new_gl_list = glGenLists(1)
    glNewList(new_gl_list, GL_COMPILE)

    # build each of the 6 faces
    # for face_index in range(6):
    #     # calculate normal and vertices for this face
    #     vertex_normal = [0.0, 0.0, 0.0]
    #     vertex_pos_options1 = [-0.1, 0.1,  0.1, -0.1]
    #     vertex_pos_options2 = [ 0.1, 0.1, -0.1, -0.1]
    #     face_index_even = ((face_index % 2) == 0)
    #     # odd and even faces point in opposite directions
    #     normal_dir = 1.0 if face_index_even else -1.0
    #     if face_index < 2:
    #         # -X and +X faces (vert positions differ in Y,Z)
    #         vertex_normal[0] = normal_dir
    #         v1i = 1
    #         v2i = 2
    #     elif face_index < 4:
    #         # -Y and +Y faces (vert positions differ in X,Z)
    #         vertex_normal[1] = normal_dir
    #         v1i = 0
    #         v2i = 2
    #     else:
    #         # -Z and +Z faces (vert positions differ in X,Y)
    #         vertex_normal[2] = normal_dir
    #         v1i = 0
    #         v2i = 1
    #
    #     vertex_pos = list(vertex_normal)
    #
    #     # Polygon (N verts) with optional normals and tex coords
    #     glBegin(GL_POLYGON)
    #     for vert_index in range(4):
    #         vertex_pos[v1i] = vertex_pos_options1[vert_index]
    #         vertex_pos[v2i] = vertex_pos_options2[vert_index]
    #         glNormal3fv(vertex_normal)
    #         glVertex3fv(vertex_pos)
    #     glEnd()
    #
    #
    # pose_matrix = pose.to_matrix()
    # glMultMatrixf(robot_matrix.in_row_order)

    # glPushMatrix();
    # glTranslatef(0.0, 0.0, -4.5)

    glBegin(GL_TRIANGLES)
    # glColor3f(0.1, 0.2, 0.3);
    # glVertex3f(-10, 0, 10)
    # glVertex3f(0, 10, 10)
    # glVertex3f(-10, 0, 10)

    glVertex3f(-10, 3, 0.0)
    glVertex3f(-10, -3, 0.0)
    glVertex3f(0, 0, 0.0)
    glEnd()
    # glPopMatrix();

    glEndList()

    return new_gl_list

def _make_pose_cube():
    new_gl_list = glGenLists(1)
    glNewList(new_gl_list, GL_COMPILE)

    # build each of the 6 faces
    for face_index in range(6):
        # calculate normal and vertices for this face
        vertex_normal = [0.0, 0.0, 0.0]
        vertex_pos_options1 = [-0.1, 0.1,  0.1, -0.1]
        vertex_pos_options2 = [ 0.1, 0.1, -0.1, -0.1]
        face_index_even = ((face_index % 2) == 0)
        # odd and even faces point in opposite directions
        normal_dir = 1.0 if face_index_even else -1.0
        if face_index < 2:
            # -X and +X faces (vert positions differ in Y,Z)
            vertex_normal[0] = normal_dir
            v1i = 1
            v2i = 2
        elif face_index < 4:
            # -Y and +Y faces (vert positions differ in X,Z)
            vertex_normal[1] = normal_dir
            v1i = 0
            v2i = 2
        else:
            # -Z and +Z faces (vert positions differ in X,Y)
            vertex_normal[2] = normal_dir
            v1i = 0
            v2i = 1

        vertex_pos = list(vertex_normal)

        # Polygon (N verts) with optional normals and tex coords
        glBegin(GL_POLYGON)
        for vert_index in range(4):
            vertex_pos[v1i] = vertex_pos_options1[vert_index]
            vertex_pos[v2i] = vertex_pos_options2[vert_index]
            glNormal3fv(vertex_normal)
            glVertex3fv(vertex_pos)
        glEnd()

    glEndList()

    return new_gl_list

def _make_unit_cube():
    """Make a unit-size cube, with normals, centered at the origin"""
    new_gl_list = glGenLists(1)
    glNewList(new_gl_list, GL_COMPILE)

    # build each of the 6 faces
    for face_index in range(6):
        # calculate normal and vertices for this face
        vertex_normal = [0.0, 0.0, 0.0]
        vertex_pos_options1 = [-1.0, 1.0,  1.0, -1.0]
        vertex_pos_options2 = [ 1.0, 1.0, -1.0, -1.0]
        face_index_even = ((face_index % 2) == 0)
        # odd and even faces point in opposite directions
        normal_dir = 1.0 if face_index_even else -1.0
        if face_index < 2:
            # -X and +X faces (vert positions differ in Y,Z)
            vertex_normal[0] = normal_dir
            v1i = 1
            v2i = 2
        elif face_index < 4:
            # -Y and +Y faces (vert positions differ in X,Z)
            vertex_normal[1] = normal_dir
            v1i = 0
            v2i = 2
        else:
            # -Z and +Z faces (vert positions differ in X,Y)
            vertex_normal[2] = normal_dir
            v1i = 0
            v2i = 1

        vertex_pos = list(vertex_normal)

        # Polygon (N verts) with optional normals and tex coords
        glBegin(GL_POLYGON)
        for vert_index in range(4):
            vertex_pos[v1i] = vertex_pos_options1[vert_index]
            vertex_pos[v2i] = vertex_pos_options2[vert_index]
            glNormal3fv(vertex_normal)
            glVertex3fv(vertex_pos)
        glEnd()

    glEndList()

    return new_gl_list


# class OpenGLWindow():
#     """A Window displaying an OpenGL viewport.
#
#     Args:
#         x (int): The initial x coordinate of the window in pixels.
#         y (int): The initial y coordinate of the window in pixels.
#         width (int): The initial height of the window in pixels.
#         height (int): The initial height of the window in pixels.
#         window_name (str): The name / title for the window.
#         is_3d (bool): True to create a Window for 3D rendering.
#     """
#     def __init__(self, x, y, width, height, window_name, is_3d):
#         self._pos = (x, y)
#         #: int: The width of the window
#         self.width = width
#         #: int: The height of the window
#         self.height = height
#         self.gl_window = None # this was set to _gl_window, not sure why since that value was unused but gl_window *is used*. Change bcz I need access
#
#         self._window_name = window_name
#         self._is_3d = is_3d

    # def close(self):
    #     sys.exit

    # def init_display(self):
    #     """Initialze the OpenGL display parts of the Window.
    #
    #     Warning:
    #         Must be called on the same thread as OpenGL (usually the main thread),
    #         and after glutInit().
    #     """
    #
    #     glutInitWindowSize(self.width, self.height)
    #     glutInitWindowPosition(*self._pos)
    #
    #     self.gl_window = glutCreateWindow(self._window_name)
    #
    #     # glutWMCloseFunc(self.close)
    #
    #     if self._is_3d:
    #         glClearColor(0, 0, 0, 0)
    #         glEnable(GL_DEPTH_TEST)
    #         glShadeModel(GL_SMOOTH)
    #
    #     glutReshapeFunc(self._reshape)
    #
    # def _reshape(self, width, height):
    #     # Called from OpenGL whenever this window is resized.
    #     self.width = width
    #     self.height = height
    #     glViewport(0, 0, width, height)


class RobotRenderFrame():
    """Minimal copy of a Robot's state for 1 frame of rendering."""
    def __init__(self, pose):
        self.pose = pose
        # self.head_angle = robot.head_angle
        # self.lift_position = robot.lift_position


class ObservableElementRenderFrame():
    """Minimal copy of a Cube's state for 1 frame of rendering."""
    def __init__(self, element):
        self.pose = element.pose
        self.is_visible = element.is_visible
        self.last_observed_time = element.last_observed_time

    @property
    def time_since_last_seen(self):
        # Equivalent of ObservableElement's method
        '''float: time since this element was last seen (math.inf if never)'''
        if self.last_observed_time is None:
            return math.inf
        return time.time() - self.last_observed_time


class CubeRenderFrame(ObservableElementRenderFrame):
    """Minimal copy of a Cube's state for 1 frame of rendering."""
    def __init__(self, cube):
        super().__init__(cube)


class FaceRenderFrame(ObservableElementRenderFrame):
    """Minimal copy of a Face's state for 1 frame of rendering."""
    def __init__(self, face):
        super().__init__(face)


class CustomObjectRenderFrame(ObservableElementRenderFrame):
    """Minimal copy of a CustomObject's state for 1 frame of rendering."""
    def __init__(self, obj, is_fixed):
        if is_fixed:
            # Not an observable, so init directly
            self.pose = obj.pose
            self.is_visible = None
            self.last_observed_time = None
        else:
            super().__init__(obj)

        self.is_fixed = is_fixed
        self.x_size_mm = obj.x_size_mm
        self.y_size_mm = obj.y_size_mm
        self.z_size_mm = obj.z_size_mm


class WorldRenderFrame():
    """Minimal copy of the World's state for 1 frame of rendering."""
    def __init__(self, pose, seen_objects):
        # world = robot.world

        self.robot_frame = RobotRenderFrame(pose)

        # self.cube_frames = []
        # for i in range(3):
        #     cube_id = objects.LightCubeIDs[i]
        #     cube = world.get_light_cube(cube_id)
        #     if cube is None:
        #         self.cube_frames.append(None)
        #     else:
        #         self.cube_frames.append(CubeRenderFrame(cube))

        # self.face_frames = []
        # for face in world._faces.values():
        #     # Ignore faces that have a newer version (with updated id)
        #     # or if they haven't been seen in a while).
        #     if not face.has_updated_face_id and (face.time_since_last_seen < 60):
        #         self.face_frames.append(FaceRenderFrame(face))


        # # TODO: catherine: bring this back
        self.custom_object_frames = []
        for obj in seen_objects:
            is_custom = isinstance(obj, objects.CustomObject)
            is_fixed = isinstance(obj, objects.FixedCustomObject)
            if is_custom or is_fixed:
                self.custom_object_frames.append(CustomObjectRenderFrame(obj, is_fixed))



class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, fig):
        ax = fig.add_subplot(111, projection='3d')
        # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        # cbar = fig.colorbar(cm.ScalarMappable(cmap=colormaps['gnuplot']), ax=ax)
        # cbar.ax.set_ylabel("Region learning potential")

        self.axes = ax

        ax.set_aspect('equal')
        fig.set_size_inches(13.5, 8.5)
        # TODO: pass this
        # fig.suptitle(f"Execution uuid: {agent.execution_uuid}")
        # ax = fig.add_subplot(111, projection='3d')
        # ax = None

        cbar = fig.colorbar(cm.ScalarMappable(cmap=colormaps['gnuplot']), ax=ax)
        cbar.ax.set_ylabel("Region learning potential")



        super().__init__(fig)

class PlotWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, fig):
        super().__init__()

        # Create canvas object
        self.canvas = MplCanvas(fig)
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)

        self.vbl = QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(toolbar)
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update) # Triggers paintEvent
        # self.timer.start(80) # ~60 FPS (1000ms / 60)
    # Correct way to render:
    # The paintEvent is called automatically by Qt
    def paintEvent(self, event):
        super().paintEvent(event)
        # self.canvas.axes.draw()
        self.canvas.draw()

    # def update_plot(self):
    #     # Drop off the first y element, append a new one.
    #     # self.ydata = self.ydata[1:] + [random.randint(0, 10)]
    #     # self.canvas.axes.cla()  # Clear the canvas.
    #     # self.canvas.axes.plot(self.xdata, self.ydata, 'r')
    #     # Trigger the canvas to update and redraw.
    #     # plt.sca(self.canvas.axes)
    #     self.canvas.draw()

class CameraViewWindow(QOpenGLWindow):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self):
        super().__init__()
        self._img_queue = collections.deque(maxlen=1)

        # self._img_queue = img_queue


    def initializeGL(self):
        glClearColor(0, 0, 0, 0)
        self._camera_view_texture = None  # type: DynamicTexture

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update) # Triggers paintGL
        self.timer.start(16) # ~60 FPS (1000ms / 60)


    def paintGL(self):
        if self._camera_view_texture is None:
            self._camera_view_texture = DynamicTexture()

        target_width = self.width()
        target_height = self.height()
        target_aspect = 320 / 240  # (Camera-feed resolution and aspect ratio)
        max_u = 1.0
        max_v = 1.0
        if (target_width / target_height) < target_aspect:
            target_height = target_width / target_aspect
            max_v *= target_height / self.height()
        elif (target_width / target_height) > target_aspect:
            target_width = target_height * target_aspect
            max_u *= target_width / self.width()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)

        # Try getting a new image if one has been added
        image = None
        try:
            image = self._img_queue.popleft()
        except IndexError:
            # no new image - queue is empty
            pass

        if image:
            # There's a new image - update the texture
            self._camera_view_texture.update(image)
        else:
            # keep using the most recent texture
            self._camera_view_texture.bind()

        # Display the image as a tri-strip with 4 vertices
        glBegin(GL_TRIANGLE_STRIP)
        # (0,0) = Top Left, (1,1) = Bottom Right
        # left, bottom
        glTexCoord2f(0.0, 1.0)
        glVertex2f(-max_u, -max_v)
        # right, bottom
        glTexCoord2f(1.0, 1.0)
        glVertex2f(max_u, -max_v)
        # left, top
        glTexCoord2f(0.0, 0.0)
        glVertex2f(-max_u, max_v)
        # right, top
        glTexCoord2f(1.0, 0.0)
        glVertex2f(max_u, max_v)
        glEnd()

        glDisable(GL_TEXTURE_2D)

        glutSwapBuffers()



class SimpleGLWindow(QOpenGLWindow):

    # def __init__(self, nav_memory_map_queue, world_frame_queue, pose_history_queue, img_queue, fig, ax):
    def __init__(self, fig):

        super().__init__()

        self.pause_event = Event()
        self.stop_event  = Event()
        self.future_pause_event = Event()
        self.learning_progress_region_colors = Event()
        self.distinct_region_colors = Event()

        # Queues from SDK thread to OpenGL thread
        self._intended_pose_history_queue = collections.deque(maxlen=1)
        self._nav_memory_map_queue = collections.deque(maxlen=1)
        self._world_frame_queue = collections.deque(maxlen=1)
        self._pose_history_queue = collections.deque(maxlen=1)

        # self._nav_memory_map_queue = nav_memory_map_queue
        # self._world_frame_queue = world_frame_queue
        # self._pose_history_queue = pose_history_queue


        # self._progress_text = ''

        self.camera_view_window = CameraViewWindow()
        self.camera_view_window.show()

        self.plot_window = PlotWindow(fig)
        self.plot_window.show()


    def initializeGL(self):
        glClearColor(0, 0, 0, 0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glShadeModel(GL_SMOOTH)

        #         glutInitContextVersion (3, 2)
        #         glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)

        #         glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)

        #         glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)

        # Load 3D objects
        _cozmo_obj = LoadedObjFile("cozmo.obj")
        self.cozmo_object = RenderableObject(_cozmo_obj)

        self.unit_cube = _make_unit_cube()
        self.pose_cube = _make_pose_cube()
        self.pose_arrow = _make_pose_arrow()
        self.origin_arrow = _make_origin_arrow()


        # Queue from OpenGL thread to SDK thread
        self._input_intent_queue = collections.deque(maxlen=1)

        # self._is_keyboard_control_enabled = False

        self._latest_world_frame = None  # type: WorldRenderFrame
        #         self._latest_pose_history = None
        self._nav_memory_map_display_list = None

        # Keyboard
        self._is_key_pressed = {}
        self._is_alt_down = False
        self._is_ctrl_down = False
        self._is_shift_down = False

        # Mouse
        self._is_mouse_down = {}
        self._mouse_pos = None  # type: util.Vector2

        # Coordinates
        self._show_coordinates = False # relative to origin
        self._show_coordinates_relative_to_robot = False

        # Pose history
        self._show_pose_history = True
        self._show_intended_pose_history = False
        self._shade_pose_by_region = False
        self._shade_pose_by_learning_progress = False
        self._shade_pose_by_age = False
        self._region_colors = []
        self._learning_progress_colors = []

        #Cozmo
        self._show_cozmo = True

        #         # Controls
        #         self._show_controls = show_viewer_controls
        #         self._instructions = '\n'.join(['W, S: Move forward, backward',
        #                                         'A, D: Turn left, right',
        #                                         'R, F: Lift up, down',
        #                                         'T, G: Head up, down',
        #                                         '',
        #                                         'LMB: Rotate camera',
        #                                         'RMB: Move camera',
        #                                         'LMB + RMB: Move camera up/down',
        #                                         'LMB + Z: Zoom camera',
        #                                         'X: same as RMB',
        #                                         'TAB: center view on robot',
        #                                         '',
        #                                         'H: Toggle help',
        #                                         'C: Show coordinate visuals',
        #                                         'P: Show pose history',
        #                                         'O: Shade pose history (opacity)',
        #                                         'B: Show Cozmo (bot)',
        #                                         'Note: Smallest square is 10mm',
        #                                         '''Note: Nav-Map child orientation is;     +---+----+---+
        #                                         | ^ | 2  | 0 |
        #                                         +---+----+---+
        #                                         | Y | 3  | 1 |
        #                                         +---+----+---+
        #                                         |   | X->|   |
        #                                         +---+----+---+'''])

        # Camera position and orientation defined by a look-at positions
        # and a pitch/and yaw to rotate around that along with a distance
        self._camera_look_at = util.Vector3(100.0, -25.0, 0.0)
        self._camera_pitch = math.radians(40)
        self._camera_yaw = math.radians(270)
        self._camera_distance = 500.0
        self._camera_pos = util.Vector3(0, 0, 0)
        self._camera_up = util.Vector3(0.0, 0.0, 1.0)
        self._calculate_camera_pos()



        # if self.plotting_window:
        #     self.plotting_window.init_display()
        #     glutDisplayFunc(self._display)


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update) # Triggers paintGL
        self.timer.start(80) # ~60 FPS (1000ms / 60)

        # self.timer.timeout.connect(self.plot_window.update_plot)

    def mousePressEvent(self, event):
        current_pos = util.Vector2(event.position().x(), event.position().y())
        if not self._mouse_pos is None:
            # print(current_pos)
            # print(self._mouse_pos)
            delta = current_pos - self._mouse_pos
            # print(f"Delta: {delta.x}, {delta.y}")

        # Check if Control key is pressed
        # if event.modifiers() & Qt.ControlModifier:
        #     print(f"Ctrl+Mouse Move: Delta X={dx}, Delta Y={dy}")

        # self.last_pos = current_pos
        self._mouse_pos = util.Vector2(event.position().x(), event.position().y())


    def mouseMoveEvent(self, event):
        last_mouse_pos = self._mouse_pos
        self._mouse_pos = util.Vector2(event.position().x(), event.position().y())
        if last_mouse_pos is None:
            # First mouse update - ignore (we need a delta of mouse positions)
            return

        MOUSE_SPEED_SCALAR = 1.0  # general scalar for all mouse movement sensitivity
        MOUSE_ROTATE_SCALAR = 0.025  # additional scalar for rotation sensitivity
        mouse_delta = (self._mouse_pos - last_mouse_pos) * MOUSE_SPEED_SCALAR


        # Equivalent to GLUT_DOWN
        if event.buttons() == Qt.MouseButton.LeftButton:
            # print(f"Left button pressed at: {event.position().x()}, {event.position().y()}")
            if self._is_key_pressed.get(b'z', False):
                # Zoom in/out
                self._camera_distance = max(0.1, self._camera_distance + mouse_delta.y)
            else:
                # print("adjusting pitch and yaw")
                # Adjust the Camera pitch and yaw
                self._camera_pitch = (self._camera_pitch - (mouse_delta.y * MOUSE_ROTATE_SCALAR))
                self._camera_yaw = (self._camera_yaw + (mouse_delta.x * MOUSE_ROTATE_SCALAR)) % (2.0 * math.pi)
                # Clamp pitch to slightyly less than pi/2 to avoid lock/errors at full up/down
                max_rotation = math.pi * 0.49
                self._camera_pitch = max(-max_rotation, min(max_rotation, self._camera_pitch))

        elif event.buttons() == Qt.MouseButton.RightButton:
            # print("Right button pressed")
            # Move forward/back and left/right
            pitch = self._camera_pitch
            yaw = self._camera_yaw
            camera_offset = util.Vector3(math.cos(yaw), math.sin(yaw), math.sin(pitch))

            heading = math.atan2(camera_offset.y, camera_offset.x)

            half_pi = math.pi * 0.5
            self._camera_look_at._x += mouse_delta.x * math.cos(heading + half_pi)
            self._camera_look_at._y += mouse_delta.x * math.sin(heading + half_pi)

            self._camera_look_at._x += mouse_delta.y * math.cos(heading)
            self._camera_look_at._y += mouse_delta.y * math.sin(heading)

        elif event.buttons() == Qt.MouseButton.RightButton and event.buttons() == Qt.MouseButton.LeftButton:
            # print("Both left and right buttons pressed")
            # Move up/down
            self._camera_look_at._z -= mouse_delta.y

    # def mouseReleaseEvent(self, event):
    #     # Equivalent to GLUT_UP
    #     if event.button() == Qt.MouseButton.LeftButton:
    #         print("Left button released")

    def keyPressEvent(self, event):
        # Equivalent to 'key' in glutKeyboardFunc
        key = event.text()
        # Equivalent to detecting modifiers
        modifiers = event.modifiers()

        # print(f"Key pressed: {key}")

        # if key == 'q':
        #     QApplication.quit()

        # # Handle special keys (like GLUT specialFunc)
        # if event.key() == Qt.Key.Key_Escape:
        #     print("Escape pressed")

        # key = self._key_byte_to_lower(key)
        # self._update_modifier_keys()
        # self._is_key_pressed[key] = True

        # if key == '9':  # Tab
        #     # Set Look-At point to current robot position
        #     world_frame = self._latest_world_frame
        #     if world_frame is not None:
        #         robot_pos = world_frame.robot_frame.pose.position
        #         self._camera_look_at.set_to(robot_pos)
        if event.key() == Qt.Key.Key_Escape:  # Escape key
            QApplication.quit()
        elif key == 'h' or key == 'H': # h or H key
            self._show_controls = not self._show_controls
        elif key == 'c' or key == 'C': # c or C key
            self._show_coordinates = not self._show_coordinates
        elif key == 'p': # p or P key
            self._show_pose_history = not self._show_pose_history
        elif key == 'i' or key == 'I': # i or I key
            self._show_intended_pose_history = not self._show_intended_pose_history
        elif key == 'o' or key == 'O': # o or O key
            self._shade_pose_by_age = not self._shade_pose_by_age
        elif key == 'q' or key == 'Q': # q or Q key
            self._shade_pose_by_region = not self._shade_pose_by_region
            self.learning_progress_region_colors.clear()
            self.distinct_region_colors.set()
        elif key == 'v' or key == 'V':
            self._shade_pose_by_learning_progress = not self._shade_pose_by_learning_progress
            self.distinct_region_colors.clear()
            self.learning_progress_region_colors.set()
        elif key == 'b' or key == 'B': # b or B key
            self._show_cozmo = not self._show_cozmo
        elif key == 'P':
            print('setting pause event')
            self.pause_event.set()
        elif key == 'e':
            self.stop_event.set()




    def paintGL(self):
        # Clear the screen and the depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        fov = 45.0
        aspect_ratio = self.width() / self.height()
        near_clip_plane = 1.0
        far_clip_plane = 5000.0
        gluPerspective(fov, aspect_ratio, near_clip_plane, far_clip_plane)

        # Switch to model matrix for rendering everything
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Add a light near the origin
        light_ambient = [1.0, 1.0, 1.0, 1.0]
        light_diffuse = [1.0, 1.0, 1.0, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
        light_pos = [0, 20, 10, 1]
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glEnable(GL_LIGHT0)


        self._calculate_camera_pos()

        gluLookAt(*self._camera_pos.x_y_z,
                  *self._camera_look_at.x_y_z,
                  *self._camera_up.x_y_z)

        # Update the latest world frame if there is a new one available
        try:
            world_frame = self._world_frame_queue.popleft()  # type: WorldRenderFrame
            self._latest_world_frame = world_frame
        except IndexError:
            world_frame = self._latest_world_frame
            pass

        if world_frame is not None:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
            glEnable(GL_NORMALIZE)  # to re-scale scaled normals

            robot_frame = world_frame.robot_frame
            robot_pose = robot_frame.pose

            #             self._draw_text(GLUT_BITMAP_HELVETICA_12, f"(x:{round(robot_pose.position.x,3)}, y:{round(robot_pose.position.y,3)})[{round(robot_pose.rotation.angle_z.degrees, 2)}°]", 0, 6)
            #             self._draw_text(GLUT_BITMAP_HELVETICA_12, self._progress_text, 0, 20)

            # self._draw_text(GLUT_BITMAP_9_BY_15, f"(x:{round(robot_pose.position.x,3)}, y:{round(robot_pose.position.y,3)})[{round(robot_pose.rotation.angle_z.degrees, 2)}°]", 0, 6)
            # self._draw_text(GLUT_BITMAP_9_BY_15, self._progress_text, 0, 20)

            for obj in world_frame.custom_object_frames:
                obj_pose = obj.pose
                if obj_pose is not None and obj_pose.is_comparable(robot_pose):
                    glPushMatrix()
                    obj_matrix = obj_pose.to_matrix()
                    glMultMatrixf(obj_matrix.in_row_order)

                    glScalef(obj.x_size_mm * 0.5,
                             obj.y_size_mm * 0.5,
                             obj.z_size_mm * 0.5)

                    # # Draw unit cube but scaled to the mm of the object
                    # glScalef(obj.x_size_mm,
                    #          obj.y_size_mm,
                    #          obj.z_size_mm)
                    # Only draw solid object for observable custom objects
                    if obj.is_fixed:
                        # fixed objects are drawn as transparent outlined boxes to make
                        # it clearer that they have no effect on vision.
                        FIXED_OBJECT_COLOR = [1.0, 0.7, 0.0, 1.0]
                        self._draw_unit_cube(FIXED_OBJECT_COLOR, False)
                    else:
                        CUSTOM_OBJECT_COLOR = [1.0, 0.3, 0.3, 1.0]
                        self._draw_unit_cube(CUSTOM_OBJECT_COLOR, True)

                    # # Draw all objects as solid, including custom objects
                    # CUSTOM_OBJECT_COLOR = [1.0, 0.3, 0.3, 1.0]
                    # self._draw_unit_cube(CUSTOM_OBJECT_COLOR, True)

                    glPopMatrix()


            # Update the latest world frame if there is a new one available
            try:
                pose_history = self._pose_history_queue.popleft()  # type: WorldRenderFrame
                self._latest_pose_history = pose_history
            except IndexError:
                pose_history = self._latest_pose_history
                pass
            if self._show_pose_history:
                if pose_history is not None:
                    for idx,past_pose in enumerate(pose_history[0]):
                        if past_pose is not None and past_pose.is_comparable(robot_pose):
                            glPushMatrix()
                            glDisable(GL_LIGHTING)  # so it shows as red from all angles

                            pose_matrix = past_pose.to_matrix()
                            glMultMatrixf(pose_matrix.in_row_order) # this appears to make it so my pose arrow is drawn with 0,0 being the pose specified

                            if self._shade_pose_by_age:
                                CUBE_OBJECT_COLOR = [1.0, 0.0, 0.0, idx/len(pose_history[0])] # red
                            elif self._shade_pose_by_region:
                                CUBE_OBJECT_COLOR = self._region_colors[idx] # whatever color for the region
                            elif self._shade_pose_by_learning_progress:
                                CUBE_OBJECT_COLOR = self._learning_progress_colors[idx]
                            else:
                                CUBE_OBJECT_COLOR = [1.0, 0.0, 0.0, 1.0] # red

                            self._draw_pose_arrow(CUBE_OBJECT_COLOR, draw_solid=True)
                            # if self._show_coordinates:
                            #     # self._draw_unit_cube([0.5, 0.5, 0.5, 1.0], True)
                            #     self._draw_unit_cube(color=[1.0, 1.0, 1.0, 0.3], draw_solid=True)
                            #     self._draw_text_on_grid(GLUT_BITMAP_9_BY_15, f"({round(pose_history[1][idx].position.x,2)}, {round(pose_history[1][idx].position.y,2)})[{round(pose_history[1][idx].rotation.angle_z.degrees, 2)}°]", 0, 0, 2)

                            glPopMatrix()

                        try:
                            intended_pose_history = self._intended_pose_history_queue.popleft()  # type: WorldRenderFrame
                            self._latest_intended_pose_history = intended_pose_history
                        except IndexError:
                            intended_pose_history = self._latest_intended_pose_history
                            pass

                        if self._show_intended_pose_history:
                            if intended_pose_history is not None:
                                for idx,past_pose in enumerate(intended_pose_history[0]):
                                    if past_pose is not None and past_pose.is_comparable(robot_pose):
                                        glPushMatrix()
                                        glDisable(GL_LIGHTING)  # so it shows as red from all angles

                                        pose_matrix = past_pose.to_matrix()
                                        glMultMatrixf(pose_matrix.in_row_order) # this appears to make it so my pose arrow is drawn with 0,0 being the pose specified

                                        if self._shade_pose_by_age:
                                            CUBE_OBJECT_COLOR = [0.0, 1.0, 0.0, idx/len(pose_history[0])] # red
                                        elif self._shade_pose_by_region:
                                            CUBE_OBJECT_COLOR = self._region_colors[idx] # whatever color for the region
                                        elif self._shade_pose_by_learning_progress:
                                            CUBE_OBJECT_COLOR = self._learning_progress_colors[idx]
                                        else:
                                            CUBE_OBJECT_COLOR = [0.0, 1.0, 0.0, 1.0] # red

                                        # glRotate(past_pose.rotation.angle_z.degrees, 0, 0)
                                        self._draw_pose_arrow(CUBE_OBJECT_COLOR, draw_solid=True)

                                        glPopMatrix()



            glDisable(GL_LIGHTING)
            if self._show_cozmo:
                self._draw_cozmo(robot_frame)


        #         if self._show_controls:
        #             self._draw_controls()

        # Draw the (translucent) nav map last so it's sorted correctly against opaque geometry
        self._draw_memory_map()

        # self._draw_origin_circle(color=[1.0, 0.0, 0.0, 1.0])
        # self._draw_unit_cube(color=[1.0, 0.0, 0.0, 1.0], draw_solid=True)
        # self._draw_origin_arrow(color=[0.16, 0.35, 1.0, 1.0])
        self._draw_origin_arrow(color=[1.0, 1.0, 1.0, 1])

        #         if self._show_coordinates:
        #             self._draw_unit_cube(color=[1.0, 1.0, 1.0, 0.3], draw_solid=True)
        #             self._draw_text_on_grid(GLUT_BITMAP_9_BY_15, '(0,0)', 0, 0)

        #             glPushMatrix() # w/ popmatrix to to save and restore the unscaled coordinate system.
        #             glTranslatef(10,0,0)
        #             self._draw_unit_cube(color=[1.0, 1.0, 1.0, 0.3], draw_solid=True)
        #             # glTranslatef(0,0,0)
        #             glPopMatrix()
        #             # glFlush()
        #             self._draw_text_on_grid(GLUT_BITMAP_9_BY_15, '(10,0)', 10, 0)
        #             #
        #             glPushMatrix()
        #             glTranslatef(-10,0,0)
        #             self._draw_unit_cube(color=[1.0, 1.0, 1.0, 0.3], draw_solid=True)
        #             glPopMatrix()
        #             self._draw_text_on_grid(GLUT_BITMAP_9_BY_15, '(-10,0)', -10, 0)

        #             # glPushMatrix()
        #             # glTranslatef(0,10,0)
        #             # self._draw_unit_cube(color=[1.0, 1.0, 1.0, 0.3], draw_solid=True)
        #             # glPopMatrix()
        #             # self._draw_text_on_grid(GLUT_BITMAP_9_BY_15, '(0,10)', 0, 10)
        #             #
        #             # glPushMatrix()
        #             # glTranslatef(0,-10,0)
        #             # self._draw_unit_cube(color=[1.0, 1.0, 1.0, 0.3], draw_solid=True)
        #             # glPopMatrix()
        #             # self._draw_text_on_grid(GLUT_BITMAP_9_BY_15, '(0,-10)', 0, -10)


        glutSwapBuffers()


    def _calculate_camera_pos(self):
        # Calculate camera position based on look-at, distance and angles
        cos_pitch = math.cos(self._camera_pitch)
        sin_pitch = math.sin(self._camera_pitch)
        cos_yaw = math.cos(self._camera_yaw)
        sin_yaw = math.sin(self._camera_yaw)
        cam_distance = self._camera_distance
        cam_look_at = self._camera_look_at

        self._camera_pos._x = cam_look_at.x + (cam_distance * cos_pitch * cos_yaw)
        self._camera_pos._y = cam_look_at.y + (cam_distance * cos_pitch * sin_yaw)
        self._camera_pos._z = cam_look_at.z + (cam_distance * sin_pitch)

    def _draw_unit_cube(self, color, draw_solid):
        glColor(color)

        if draw_solid:
            ambient_color = [color[0]*0.1, color[1]*0.1, color[2]*0.1, 1.0]
        else:
            ambient_color = color
        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_color)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
        glMaterialfv(GL_FRONT, GL_SPECULAR,  color)

        glMaterialfv(GL_FRONT, GL_SHININESS, 10.0);

        if draw_solid:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glCallList(self.unit_cube)

    def _draw_memory_map(self):
        # Update the renderable map if new data is available, and
        # render the latest map received.
        new_nav_memory_map = None
        try:
            new_nav_memory_map = self._nav_memory_map_queue.popleft()
        except IndexError:
            # no new nav map - queue is empty
            pass

        # Rebuild the renderable map if it has changed
        if new_nav_memory_map is not None:
            cen = new_nav_memory_map.center
            half_size = new_nav_memory_map.size * 0.5

            if self._nav_memory_map_display_list is None:
                self._nav_memory_map_display_list = glGenLists(1)
            glNewList(self._nav_memory_map_display_list, GL_COMPILE)

            glPushMatrix()
            # draw center of nav memory map. Sneaking in here so it doesn't flash (due to render order)...there is probably a better place elsewhere
            glColor([1.0, 1.0, 1.0, 0.3])
            posx = cen.x
            posy = cen.y
            sides = 32
            radius = 10
            glBegin(GL_POLYGON)
            for i in range(100):
                cosine= radius * math.cos(i*2*math.pi/sides) + posx
                sine  = radius * math.sin(i*2*math.pi/sides) + posy
                glVertex2f(cosine,sine)
            glEnd()

            # draw lines of nav memory map (the grid)
            color_light_gray = (0.65, 0.65, 0.65)
            glColor3f(*color_light_gray)
            glBegin(GL_LINE_STRIP)
            glVertex3f(cen.x + half_size, cen.y + half_size, cen.z)  # TL
            glVertex3f(cen.x + half_size, cen.y - half_size, cen.z)  # TR
            glVertex3f(cen.x - half_size, cen.y - half_size, cen.z)  # BR
            glVertex3f(cen.x - half_size, cen.y + half_size, cen.z)  # BL
            glVertex3f(cen.x + half_size, cen.y + half_size,
                       cen.z)  # TL (close loop)
            glEnd()

            def color_for_content(content):
                nct = nav_memory_map.NodeContentTypes
                colors = {nct.Unknown.id: (0.3, 0.3, 0.3),         # dark gray
                          nct.ClearOfObstacle.id: (0.0, 1.0, 0.0), # green
                          nct.ClearOfCliff.id: (0.0, 0.5, 0.0),    # dark green
                          nct.ObstacleCube.id: (1.0, 0.0, 0.0),    # red
                          nct.ObstacleCharger.id: (1.0, 0.5, 0.0), # orange
                          nct.Cliff.id: (0.0, 0.0, 0.0),           # black
                          nct.VisionBorder.id: (1.0, 1.0, 0.0)     # yellow
                          }

                col = colors.get(content.id)
                if col is None:
                    logger.error("Unhandled content type %s" % str(content))
                    col = (1.0, 1.0, 1.0)  # white
                return col

            fill_z = cen.z - 0.4

            def _recursive_draw(grid_node: nav_memory_map.NavMemoryMapGridNode):
                if grid_node.children is not None:
                    for child in grid_node.children:
                        _recursive_draw(child)
                else:
                    # leaf node - render as a quad
                    map_alpha = 0.5
                    cen = grid_node.center
                    half_size = grid_node.size * 0.5

                    # Draw outline
                    glColor4f(*color_light_gray, 1.0)  # fully opaque
                    glBegin(GL_LINE_STRIP)
                    glVertex3f(cen.x + half_size, cen.y + half_size, cen.z)
                    glVertex3f(cen.x + half_size, cen.y - half_size, cen.z)
                    glVertex3f(cen.x - half_size, cen.y - half_size, cen.z)
                    glVertex3f(cen.x - half_size, cen.y + half_size, cen.z)
                    glVertex3f(cen.x + half_size, cen.y + half_size, cen.z)
                    glEnd()

                    # Draw filled contents
                    glColor4f(*color_for_content(grid_node.content), map_alpha)
                    glBegin(GL_TRIANGLE_STRIP)
                    glVertex3f(cen.x + half_size, cen.y + half_size, fill_z)
                    glVertex3f(cen.x + half_size, cen.y - half_size, fill_z)
                    glVertex3f(cen.x - half_size, cen.y + half_size, fill_z)
                    glVertex3f(cen.x - half_size, cen.y - half_size, fill_z)
                    glEnd()

            _recursive_draw(new_nav_memory_map.root_node)

            glPopMatrix()
            glEndList()
        else:
            # The source data hasn't changed - keep using the same call list
            pass

        if self._nav_memory_map_display_list is not None:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_BLEND)
            glPushMatrix()
            glCallList(self._nav_memory_map_display_list)
            glPopMatrix()

    def _draw_cozmo(self, robot_frame):
        if self.cozmo_object is None:
            return
        robot_pose = robot_frame.pose
        # print(f"Robot pose: {robot_pose}")
        robot_head_angle = robot.MIN_HEAD_ANGLE #robot_frame.head_angle
        robot_lift_position = LiftPosition(height=util.distance_mm(robot.MIN_LIFT_HEIGHT_MM)) #robot_frame.lift_position

        # Angle of the lift in the object's initial default pose.
        LIFT_ANGLE_IN_DEFAULT_POSE = -11.36

        robot_matrix = robot_pose.to_matrix()
        head_angle = robot_head_angle.degrees
        # Get the angle of Cozmo's lift for rendering - we subtract the angle
        # of the lift in the default pose in the object, and apply the inverse
        # rotation
        lift_angle = -(robot_lift_position.angle.degrees - LIFT_ANGLE_IN_DEFAULT_POSE)

        glPushMatrix()
        glEnable(GL_LIGHTING)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        glMultMatrixf(robot_matrix.in_row_order)
        # Scale the obj mesh dimensions from cm to mm (we are treating 1 mm as the base unit for this world rendering)
        robot_scale_amt = 10.0  # cm to mm
        # glScalef function produces a general scaling along the x, y, and z axes.
        # The three arguments indicate the desired scale factors along each of the three axes.
        glScalef(robot_scale_amt, robot_scale_amt, robot_scale_amt)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Pivot offset for where the fork rotates around itself
        FORK_PIVOT_X = 3.0
        FORK_PIVOT_Z = 3.4

        # Offset for the axel that the upper arm rotates around.
        UPPER_ARM_PIVOT_X = -3.73
        UPPER_ARM_PIVOT_Z = 4.47

        # Offset for the axel that the lower arm rotates around.
        LOWER_ARM_PIVOT_X = -3.74
        LOWER_ARM_PIVOT_Z = 3.27

        # Offset for the pivot that the head rotates around.
        HEAD_PIVOT_X = -1.1
        HEAD_PIVOT_Z = 4.75

        # Render the static body meshes - first the main body:
        glCallList(self.cozmo_object.meshes["body_geo"])
        # Render the left treads and wheels
        glCallList(self.cozmo_object.meshes["trackBase_L_geo"])
        glCallList(self.cozmo_object.meshes["wheel_BL_geo"])
        glCallList(self.cozmo_object.meshes["wheel_FL_geo"])
        glCallList(self.cozmo_object.meshes["tracks_L_geo"])
        # Render the right treads and wheels
        glCallList(self.cozmo_object.meshes["trackBase_R_geo"])
        glCallList(self.cozmo_object.meshes["wheel_BR_geo"])
        glCallList(self.cozmo_object.meshes["wheel_FR_geo"])
        glCallList(self.cozmo_object.meshes["tracks_R_geo"])

        # Render the fork at the front (but not the arms)
        glPushMatrix()
        # The fork rotates first around upper arm (to get it to the correct position).
        glTranslatef(UPPER_ARM_PIVOT_X, 0.0, UPPER_ARM_PIVOT_Z)
        glRotatef(lift_angle, 0, 1, 0)
        glTranslatef(-UPPER_ARM_PIVOT_X, 0.0, -UPPER_ARM_PIVOT_Z)
        # The fork then rotates back around itself as it always hangs vertically.
        glTranslatef(FORK_PIVOT_X, 0.0, FORK_PIVOT_Z)
        glRotatef(-lift_angle, 0, 1, 0)
        glTranslatef(-FORK_PIVOT_X, 0.0, -FORK_PIVOT_Z)
        # Render
        glCallList(self.cozmo_object.meshes["fork_geo"])
        glPopMatrix()

        # Render the upper arms:
        glPushMatrix()
        # Rotate the upper arms around the upper arm joint
        glTranslatef(UPPER_ARM_PIVOT_X, 0.0, UPPER_ARM_PIVOT_Z)
        glRotatef(lift_angle, 0, 1, 0)
        glTranslatef(-UPPER_ARM_PIVOT_X, 0.0, -UPPER_ARM_PIVOT_Z)
        # Render
        glCallList(self.cozmo_object.meshes["uprArm_L_geo"])
        glCallList(self.cozmo_object.meshes["uprArm_geo"])
        glPopMatrix()

        # Render the lower arms:
        glPushMatrix()
        # Rotate the lower arms around the lower arm joint
        glTranslatef(LOWER_ARM_PIVOT_X, 0.0, LOWER_ARM_PIVOT_Z)
        glRotatef(lift_angle, 0, 1, 0)
        glTranslatef(-LOWER_ARM_PIVOT_X, 0.0, -LOWER_ARM_PIVOT_Z)
        # Render
        glCallList(self.cozmo_object.meshes["lwrArm_L_geo"])
        glCallList(self.cozmo_object.meshes["lwrArm_R_geo"])
        glPopMatrix()

        # Render the head:
        glPushMatrix()
        # Rotate the head around the pivot
        glTranslatef(HEAD_PIVOT_X, 0.0, HEAD_PIVOT_Z)
        glRotatef(-head_angle, 0, 1, 0)
        glTranslatef(-HEAD_PIVOT_X, 0.0, -HEAD_PIVOT_Z)
        # Render all of the head meshes
        glCallList(self.cozmo_object.meshes["head_geo"])
        # Screen
        glCallList(self.cozmo_object.meshes["backScreen_mat"])
        glCallList(self.cozmo_object.meshes["screenEdge_geo"])
        glCallList(self.cozmo_object.meshes["overscan_1_geo"])
        # Eyes
        glCallList(self.cozmo_object.meshes["eye_L_geo"])
        glCallList(self.cozmo_object.meshes["eye_R_geo"])
        # Eyelids
        glCallList(self.cozmo_object.meshes["eyeLid_R_top_geo"])
        glCallList(self.cozmo_object.meshes["eyeLid_L_top_geo"])
        glCallList(self.cozmo_object.meshes["eyeLid_L_btm_geo"])
        glCallList(self.cozmo_object.meshes["eyeLid_R_btm_geo"])
        # Face cover (drawn last as it's translucent):
        glCallList(self.cozmo_object.meshes["front_Screen_geo"])
        glPopMatrix()

        glDisable(GL_LIGHTING)
        glPopMatrix()

    def _draw_origin_arrow(self, color):
        glColor(color)

        ambient_color = [color[0]*0.1, color[1]*0.1, color[2]*0.1, 1.0]

        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_color)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
        glMaterialfv(GL_FRONT, GL_SPECULAR,  color)

        glMaterialfv(GL_FRONT, GL_SHININESS, 10.0);


        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


        glCallList(self.origin_arrow)


    def _draw_pose_arrow(self, color, draw_solid):
        # glDisable(GL_LIGHTING) # so it shows as red from all angles
        glColor(color)

        if draw_solid:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_color)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  color)


        glMaterialfv(GL_FRONT, GL_SHININESS, 10.0)


        glCallList(self.pose_arrow)


    def _draw_pose_cube(self, color, draw_solid):
        glColor(color)

        if draw_solid:
            ambient_color = [color[0]*0.1, color[1]*0.1, color[2]*0.1, 1.0]
        else:
            ambient_color = color
        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_color)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
        glMaterialfv(GL_FRONT, GL_SPECULAR,  color)

        glMaterialfv(GL_FRONT, GL_SHININESS, 10.0);

        if draw_solid:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glCallList(self.pose_cube)


    def _draw_unit_cube(self, color, draw_solid):
        glColor(color)

        if draw_solid:
            ambient_color = [color[0]*0.1, color[1]*0.1, color[2]*0.1, 1.0]
        else:
            ambient_color = color
        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_color)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
        glMaterialfv(GL_FRONT, GL_SPECULAR,  color)

        glMaterialfv(GL_FRONT, GL_SHININESS, 10.0);

        if draw_solid:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glCallList(self.unit_cube)