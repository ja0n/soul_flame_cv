vert_shader = '''
#version 330 core

in vec2 vert;
in vec2 texcoord;
uniform float     time;                 // shader playback time (in seconds)
out vec2 uvs;

void main() {
    uvs = texcoord;
    gl_Position = vec4(vert, 0.0, 1.0);
}
'''

frag_shader = '''
#version 330 core

uniform sampler2D tex;

in vec2 uvs;
out vec4 f_color;

void main() {
    vec4 frag_color = texture(tex, uvs);
    f_color = vec4(frag_color.r, frag_color.g, frag_color.b * 1.7, 1.0);
}
'''
