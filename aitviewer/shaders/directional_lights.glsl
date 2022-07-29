struct DirLight {
    vec3 pos;
    vec3 color;
    float intensity_ambient;
    float intensity_diffuse;
    bool shadow_enabled;
    mat4 matrix;
};

#define NR_DIR_LIGHTS 2
uniform DirLight dirLights[NR_DIR_LIGHTS];

uniform float diffuse_coeff;
uniform float ambient_coeff;

vec3 directionalLight(DirLight dirLight, vec3 color, vec3 fragPos, vec3 normal, float shadow) {
    // Ambient
    vec3 ambient = dirLight.intensity_ambient * dirLight.color * ambient_coeff;

    // Diffuse
    vec3 lightDir = normalize(dirLight.pos - fragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diffuse_coeff * diff * dirLight.intensity_diffuse * dirLight.color;

    // Specular
//    vec3 viewDir = normalize(viewPos - fragPos);
//    vec3 reflectDir = reflect(-lightDir, normal);
//    float spec = pow(max(dot(viewDir, reflectDir), 0.0), dirLight.shininess);
//    vec3 specular = dirLight.specular * spec * dirLight.color;

    return (ambient + (1.0 - shadow)*diffuse) * color;
}
