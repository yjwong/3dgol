#version 330

in vec4 f_color;
out vec4 out_color;

void main() {
	out_color = vec4(f_color.rgb, 1.0f);
}

