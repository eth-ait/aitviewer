vec3 distanceToEdge(vec4 v0, vec4 v1, vec4 v2, vec2 win_size) {
    // From "Single-Pass Wireframe Rendering".
    vec2 p0 = win_size * v0.xy/v0.w;
    vec2 p1 = win_size * v1.xy/v1.w;
    vec2 p2 = win_size * v2.xy/v2.w;
    vec2 pp0 = p2-p1;
    vec2 pp1 = p2-p0;
    vec2 pp2 = p1-p0;
    float area = abs(pp1.x * pp2.y - pp1.y * pp2.x);
    vec3 dist = vec3(area / length(pp0),
                     area / length(pp1),
                     area / length(pp2));
    return dist;
}