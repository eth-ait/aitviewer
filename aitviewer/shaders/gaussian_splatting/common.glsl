// Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

struct Splat {
    vec3 position;
    vec3 color;
    float opacity;
    vec3 scale;
    vec4 rotation;
};

struct SplatData {
    vec3 color;
    float opacity;
    vec3 scale;
    float _padding;
    vec4 rotation;
};

struct SplatView {
    vec4 position;
    vec2 axis1;
    vec2 axis2;
    vec4 color;
};
