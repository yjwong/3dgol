#version 330

uniform vec4 mat_ambient;
uniform vec4 mat_diffuse;
uniform vec4 mat_specular;
uniform float mat_shininess;
uniform vec4 mat_emission;

uniform vec4 light_ambient;
uniform vec4 light_diffuse;
uniform vec4 light_specular;
uniform vec3 light_position;

in vec3 f_position;
in vec4 f_color;
in vec3 f_normal;

out vec4 out_color;

void main() {
	// Some useful eye-space vectors.
	vec3 ec_nnormal = normalize(f_normal);
	vec3 ec_view_vec = -normalize(f_position);

	// Perform some vector computation for the lights.
	ec_nnormal = gl_FrontFacing ? ec_nnormal : -ec_nnormal;
	vec3 ec_light_pos = light_position;
	vec3 ec_light_vec = normalize(ec_light_pos - f_position);
	vec3 half_vector = normalize(ec_light_vec + ec_view_vec);

	float N_dot_L = max(0.0f, dot(ec_nnormal, ec_light_vec));
	float N_dot_H = max(0.0f, dot(ec_nnormal, half_vector));

	// Compute the specular factor.
	float specular_factor = (N_dot_H == 0.0f) ? 0.0f : pow(N_dot_H, mat_shininess);

	// Compute the out color using Phong reflection model.
	out_color = vec4(f_color.rgb, 1.0f) * (
				vec4(0.1f, 0.1f, 0.1f, 1.0f) +
				mat_ambient * light_ambient +
				mat_diffuse * light_diffuse * N_dot_L
			) +	light_specular * mat_specular * specular_factor;
}

