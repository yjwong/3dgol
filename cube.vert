#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform int count;
uniform vec3 field_position;

out vec4 f_color;
out vec3 f_position;

void main(void) {
	f_color = vec4(color, 1.0f);
	f_position = position;

	// Scaling matrix to scale the cube.
	mat4 scale_transform = mat4(
		vec4(1.0f / count, 0.0f, 0.0f, 0.0f),
		vec4(0.0f, 1.0f / count, 0.0f, 0.0f),
		vec4(0.0f, 0.0f, 1.0f / count, 0.0f),
		vec4(0.0f, 0.0f, 0.0f, 1.0f)
	);

	// Translation matrix to translate cube to correct location.
	mat4 translate_transform = mat4(
		vec4(1.0f, 0.0f, 0.0f, 0.0f),
		vec4(0.0f, 1.0f, 0.0f, 0.0f),
		vec4(0.0f, 0.0f, 1.0f, 0.0f),
		vec4(
			field_position.x * 2.0f - (count - 1.0f),
			field_position.y * 2.0f - (count - 1.0f),
			field_position.z * 2.0f - (count - 1.0f), 1.0f)
	);

	mat4 transformed_model = model * scale_transform * translate_transform;
	gl_Position = projection * view * transformed_model * vec4(position, 1.0f);
}

