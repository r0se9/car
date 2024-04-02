{@}Background.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uBaseColor;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform float uNoiseScale;
uniform float uNoiseSpeed;
uniform float uVisible;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(rgb2hsv.fs)

vec3 dither(vec3 color) {
    float grid_position = random(gl_FragCoord.xy);
    vec3 dither_shift_RGB = vec3(0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0);
    dither_shift_RGB = mix(2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position);
    return color + dither_shift_RGB;
}

void main() {
    vec2 uv = vUv + getNoise(vUv, time * 0.5) * 0.02;
    uv += vec2(time * 0.05);
    float noise = cnoise(vec3(uv * uNoiseScale, time * uNoiseSpeed * 0.6));
    float noise2 = cnoise(vec3(uv * uNoiseScale * 0.8, time * uNoiseSpeed * 0.8));

    vec3 color = uBaseColor;

    color = mix(color, uColor1, max(0.0, noise)*0.06*uVisible);
    color = mix(color, uColor2, max(0.0, -noise)*0.06*uVisible);
    color = mix(color, uColor3, max(0.0, -noise2)*0.06*uVisible);

    color = dither(color);

    gl_FragColor.rgb = color;//color;
    gl_FragColor.a = 1.0;
}
{@}BackgroundParticles.glsl{@}#!ATTRIBUTES

attribute vec4 random;

#!UNIFORMS

uniform sampler2D tParticle;

#!VARYINGS

varying vec3 vRandom1;
varying vec4 vRandom2;

#!SHADER: Vertex

void main() {
    vRandom1 = position;
    vRandom2 = random;

    float span = 10.0;

    vec3 pos = vec3(
        mix(-span, span, position.x),
        mix(-span, span, position.y),
        mix(-3.0 * span, 0.0, position.z)
    );

    pos.x += sin(time * 0.2 * random.x + random.y * 6.28) * mix(span * 0.1, span * 0.5, random.z);
    pos.y += sin(time * 0.2 * random.w + random.x * 6.28) * mix(span * 0.1, span * 0.5, random.y);
    pos.z += sin(time * 0.2 * random.z + random.w * 6.28) * mix(span * 0.1, span * 0.5, random.x);

    vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mvPos;
    gl_PointSize = mix(0.2, 1.0, random.x) * 500.0 / length(mvPos.xyz);
}

#!SHADER: Fragment

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

void main() {
    vec2 uv = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y);
    uv -= 0.5;
    rotate2D(uv, mix(0.0, 6.26, vRandom2.x));
    uv.x *= mix(1.0, 1.5, vRandom2.y);
    uv.y *= mix(1.0, 1.5, vRandom2.w);
    uv += 0.5;

    float alpha = mix(0.05, 0.1, vRandom2.y);
    alpha *= max(0.0, mix(-0.5, 1.0, sin(time * 0.5 * vRandom2.x + vRandom2.w * 6.28) * 0.5 + 0.5));

    gl_FragColor.rgb = texture2D(tParticle, uv).rgb;
    gl_FragColor.a = alpha;
}
{@}CarBackground.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tBackground;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)

void main() {
    vec3 color = texture2D(tBackground, vUv).rgb;

    gl_FragColor.rgb = color;
    gl_FragColor.a = 1.0;
}
{@}CarBackgroundParticles.glsl{@}#!ATTRIBUTES

attribute vec4 random;

#!UNIFORMS

uniform sampler2D tParticle;
uniform vec3 uCenterPosition;
uniform float uMaxRange;
uniform float uMaxHeight;

#!VARYINGS

varying vec3 vRandom1;
varying vec4 vRandom2;

#!SHADER: Vertex

#require(range.glsl)

vec2 frag_coord(vec4 glPos) {
    return ((glPos.xyz / glPos.w) * 0.5 + 0.5).xy;
}

float when_gt(float x, float y) {
  return max(sign(x - y), 0.0);
}

float when_lt(float x, float y) {
  return max(sign(y - x), 0.0);
}

float treadmill(float v, float l) {

    // Make range -l > l to 0 > 1
    float n = (v + l) / (2.0 * l);

    n = mod(n, 1.0);

    // Set range back
    n = n * 2.0 * l - l;

    return n;
}

void main() {
    vRandom1 = position;
    vRandom2 = random;

    float span = 8.0;

    vec3 pos = vec3(
        mix(-span, span, position.x),
        mix(0.0, span * 0.5, position.y),
        mix(-span, span, position.z)
    );

    pos.x += sin(time * 0.2 * random.x + random.y * 6.28) * mix(span * 0.1, span * 0.5, random.z);
    pos.y += sin(time * 0.2 * random.w + random.x * 6.28) * mix(span * 0.1, span * 0.5, random.y);
    pos.z += sin(time * 0.2 * random.z + random.w * 6.28) * mix(span * 0.1, span * 0.5, random.x);

    // Keep particles near car
    vec3 dist = pos + uCenterPosition;
    pos.x = treadmill(dist.x, uMaxRange);
    pos.z = treadmill(dist.z, uMaxRange);


    vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mvPos;
    gl_PointSize = 0.5 * mix(0.2, 1.0, random.x) * 200.0 / length(mvPos.xyz);
}

#!SHADER: Fragment

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

void main() {
    vec2 uv = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y);
    uv -= 0.5;
    rotate2D(uv, mix(0.0, 6.26, vRandom2.x));
    uv.x *= mix(1.0, 1.5, vRandom2.y);
    uv.y *= mix(1.0, 1.5, vRandom2.w);
    uv += 0.5;

    float alpha = mix(0.1, 1.0, vRandom2.y);
    alpha *= max(0.0, mix(-0.5, 1.0, sin(time * 0.5 * vRandom2.x + vRandom2.w * 6.28) * 0.5 + 0.5));

    gl_FragColor.rgb = texture2D(tParticle, uv).rgb;
    gl_FragColor.a = alpha;
}
{@}CarColor.fs{@}    // Mix of all colors
    float index = mod(vLineIndex + 0.1, 3.0);
    // vec3 color = vec3(0.5);
    vec3 color = uColor1;
    color = mix(color, uColor2, smoothstep(0.5, 0.51, index));
    color = mix(color, uColor3, smoothstep(1.5, 1.51, index));

    // Single color isolations
    color = mix(color, uColor1, uColorIsolate.x);
    color = mix(color, uColor2, uColorIsolate.y);
    color = mix(color, uColor3, uColorIsolate.z);

    color = mix(color, vec3(0.5), 0.4); // mute the colors

    float colorShine = sin(vUv2.x * 1.2 - time * 0.1 - uRoll * 1.0 + vRandom * 6.41);
    color = mix(color, vec3(1.0), colorShine);
{@}CarCopyBody.fs{@}uniform sampler2D tCarLines;
uniform sampler2D tDiffuse;

varying vec2 vUv;

void main() {
    gl_FragColor.rgb = texture2D(tDiffuse, vUv).rgb;
    gl_FragColor.rgb += texture2D(tCarLines, vUv).rgb;
    gl_FragColor.a = 1.0;
}
{@}CarFloor.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMap;
uniform float uGrid;
uniform float uDispAmount;
uniform vec3 uCenterPosition;
uniform float uSpeed;
uniform float uTime;
uniform vec3 uColor;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    vec2 floorUv = vUv * uGrid;
    floorUv += vec2(uCenterPosition.x, uCenterPosition.z) * uDispAmount;
    vec4 particles = texture2D(tMap, floorUv) * 1.5;

    float dist = length(vUv - 0.5);

    float baseTime = uTime * 0.4;

    float t = mod(baseTime, 1.0);
    float ripples = smoothstep(t + 0.06, t, dist) * smoothstep(t - 0.06, t, dist);
    float t2 = mod(baseTime - 0.13, 1.0);
    ripples += smoothstep(t2 + 0.05, t2, dist) * smoothstep(t2 - 0.05, t2, dist) * 0.85;

    particles *= mix(0.25, 0.5, abs(uSpeed)) + 0.5 * min(1.0, ripples) * 2.0;

    float fadeEdge = smoothstep(0.45, 0.2, dist);
    particles *= fadeEdge * 0.5 + uSpeed*0.3;

    gl_FragColor.rgb = mix(particles.rgb, particles.r * uColor, smoothstep(0.12, 0.22, dist));
    gl_FragColor.a = particles.a;
}
{@}CarFlowBody.glsl{@}#!ATTRIBUTES

attribute vec3 previous;
attribute vec3 next;
attribute float side;
attribute float width;
attribute float lineIndex;
attribute vec2 uv2;

#!UNIFORMS

uniform float uOpacity;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec3 uColorIsolate;
uniform float uTransition;
uniform float uSide;
uniform float uDisable;
uniform vec3 uDisabledColor;

uniform sampler2D tMask;
uniform float uRoll;
uniform float uTurn;
uniform float uSpeed;
uniform float uAddedTime;

#!VARYINGS
varying float vLineIndex;
varying vec2 vUv;
varying vec2 vUv2;
varying float vRandom;
varying float vOpa;


#!SHADER: Vertex

#require(range.glsl)
#require(simplenoise.glsl)

vec2 when_eq(vec2 x, vec2 y) {
    return 1.0 - abs(sign(x - y));
}

float when_lt(float x, float y) {
    return max(sign(y - x), 0.0);
}

vec2 fix(vec4 i, float aspect) {
    vec2 res = i.xy / i.w;
    res.x *= aspect;
    return res;
}

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

void randomCurveStarts(inout vec3 pos, float r) {
    float strength = smoothstep(1.5, 30.0, pos.z * pos.z);
    // pos.x += uTurn * (z * z * 0.05) * strength - uTurn * 0.25 * strength;
    float rand = sin((r * 0.5 + 0.5) * time * 0.2 + uRoll * 0.8 + r * 3.14) * (r * 0.5 + 0.5) * 0.3;
    pos.y += rand * strength * 1.2;
    pos.x += rand * strength * 0.5;
}

void turnCurves(inout vec3 pos) {
    float z = pos.z;
    float strength = smoothstep(0.9, 4.0, pos.z) + 0.2;
    pos.x += uTurn * (z * z * 0.05) * strength - uTurn * 0.25 * strength;
}

const float maxWidth = 2.5;

void main() {
    float aspect = resolution.x / resolution.y;

    vUv = uv;
    vUv2 = uv2;
    vLineIndex = lineIndex;
    vRandom = random(vec2(lineIndex + 2.3));

    vec3 pos = position;
    vec3 prevPos = previous;
    vec3 nextPos = next;

    // randomise starts of curves
    randomCurveStarts(pos, vRandom);
    randomCurveStarts(prevPos, vRandom);
    randomCurveStarts(nextPos, vRandom);

    // Turn curves to face direction
    turnCurves(pos);
    turnCurves(prevPos);
    turnCurves(nextPos);

    mat4 m = projectionMatrix * modelViewMatrix;
    vec4 finalPosition = m * vec4(pos, 1.0);
    vec4 pPos = m * vec4(prevPos, 1.0);
    vec4 nPos = m * vec4(nextPos, 1.0);
    vec2 currentP = fix(finalPosition, aspect);
    vec2 prevP = fix(pPos, aspect);
    vec2 nextP = fix(nPos, aspect);

    float w = 0.1 * width;

    w *= 1.0 - 0.5 * (sin(vUv2.x * 1.3 * 1.0 - time * 2.93 - uRoll * 1.0 + vRandom * 2.31) * 0.5 + 0.5);
    w *= 1.0 - 0.3 * (sin(vUv2.x * 1.3 * 3.0 - time * 1.34 - uRoll * 1.0 + vRandom * 0.87) * 0.5 + 0.5);

    vec2 dirNC = normalize(currentP - prevP);
    vec2 dirPC = normalize(nextP - currentP);
    vec2 dir1 = normalize(currentP - prevP);
    vec2 dir2 = normalize(nextP - currentP);
    vec2 dirF = normalize(dir1 + dir2);
    vec2 dirM = mix(dirPC, dirNC, when_eq(nextP, currentP));
    vec2 dir = mix(dirF, dirM, clamp(when_eq(nextP, currentP) + when_eq(prevP, currentP), 0.0, 1.0));
    vec2 normal = vec2(-dir.y, dir.x);
    normal.x /= aspect;

    float lW = mix(-5.0, 5.0, uTransition);
    float direction = mix(-1.0, 1.0, uSide);
    float ppp = (position.z - position.y) * direction;
    float ww = smoothstep(lW - 1.5, lW - 0.5, ppp) * maxWidth - smoothstep(lW - 0.5, lW + 1.5, ppp) * maxWidth * 2.0;
    float lTransform = mix(1.0, 0.0, uSide);
    w += ww * w * lTransform;
    vOpa = 1.0 - smoothstep(lW + 0.5, lW + 1.5, ppp);

    normal *= 0.5 * w;
    finalPosition.xy += normal * side;
    gl_Position = finalPosition;
}

#!SHADER: Fragment

#require(rgb2hsv.fs)

void main() {
    vec2 uvMask = vUv;
    uvMask.x += uRoll;
    uvMask.x += time * 0.1;
    vec3 mask = texture2D(tMask, uvMask).rgb;

    #require(CarColor.fs)

    gl_FragColor.rgb = mix(color, uDisabledColor, uDisable);

    gl_FragColor.a = max(max(mask.r, mask.g), mask.b);

    // fade out every now and again
    gl_FragColor.a *= smoothstep(-0.8, -0.5, sin(vUv2.x * 0.5 - time * 1.0 - uAddedTime - uRoll * 2.0 + vRandom * 4.32));
    gl_FragColor.a *= smoothstep(0.5, 0.3, sin(vUv2.x * 0.7 - time * 0.6 * vRandom - uAddedTime - uRoll * 1.0 + vRandom * 2.91));
    // gl_FragColor.a *= smoothstep(-0.8, -0.5, sin(vUv2.x * 0.5 - uRoll * 2.0 + vRandom * 4.32));
    // gl_FragColor.a *= smoothstep(0.5, 0.3, sin(vUv2.x * 0.7 * vRandom - uRoll * 1.0 + vRandom * 2.91));

    // Reduce curves to just on car while not driving
    // uSpeed
    float minLimit = mix(0.35, 0.0, abs(uSpeed));
    float maxLimit = mix(0.65, 1.0, abs(uSpeed));
    gl_FragColor.a *= smoothstep(minLimit - 0.05, minLimit, vUv.x);
    gl_FragColor.a *= smoothstep(maxLimit + 0.05, maxLimit, vUv.x);

    gl_FragColor.a *= uOpacity * vOpa;
}
{@}CarFlowGround.glsl{@}#!ATTRIBUTES

attribute vec3 previous;
attribute vec3 next;
attribute float side;
attribute float width;
attribute float lineIndex;
attribute vec2 uv2;

#!UNIFORMS

uniform float uOpacity;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec3 uColorIsolate;
uniform float uTransition;
uniform float uSide;
uniform float uDisable;
uniform float uAddedTime;
uniform vec3 uDisabledColor;

uniform sampler2D tMask;
uniform float uRoll;
uniform float uTurn;

#!VARYINGS
varying float vLineIndex;
varying vec2 vUv;
varying vec2 vUv2;
varying float vRandom;
varying float vOpa;


#!SHADER: Vertex

#require(range.glsl)
#require(simplenoise.glsl)

vec2 when_eq(vec2 x, vec2 y) {
    return 1.0 - abs(sign(x - y));
}

float when_lt(float x, float y) {
    return max(sign(y - x), 0.0);
}

vec2 fix(vec4 i, float aspect) {
    vec2 res = i.xy / i.w;
    res.x *= aspect;
    return res;
}

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

void turnCurves(inout vec3 pos) {
    // vec2 offset = vec2(0.82, 1.27);
    // offset.x *= mix(1.0, -1.0, when_lt(pos.x, 0.0));
    // float turn = uTurn * 0.5;
    // turn *= when_lt(-pos.z, 0.0);

    // pos.xz -= offset;
    // rotate2D(pos.xz, turn);
    // pos.xz += offset;

    float z = pos.z;
    pos.x += uTurn * (z * z * 0.15) - uTurn * 0.25;
}

const float maxWidth = 2.5;

void main() {
    float aspect = resolution.x / resolution.y;

    vUv = uv;
    vUv2 = uv2;
    vLineIndex = lineIndex;
    vRandom = random(vec2(lineIndex));

    vec3 pos = position;
    vec3 prevPos = previous;
    vec3 nextPos = next;

    // Turn curves to face direction
    turnCurves(pos);
    turnCurves(prevPos);
    turnCurves(nextPos);

    mat4 m = projectionMatrix * modelViewMatrix;
    vec4 finalPosition = m * vec4(pos, 1.0);
    vec4 pPos = m * vec4(prevPos, 1.0);
    vec4 nPos = m * vec4(nextPos, 1.0);
    vec2 currentP = fix(finalPosition, aspect);
    vec2 prevP = fix(pPos, aspect);
    vec2 nextP = fix(nPos, aspect);

    float w = 0.1 * width;

    w *= 1.0 - 0.5 * (sin(vUv.x * 6.28 * 1.0 - time * 2.93 - uRoll * 3.0 + vRandom * 2.31) * 0.5 + 0.5);
    w *= 1.0 - 0.3 * (sin(vUv.x * 6.28 * 3.0 - time * 1.34 - uRoll * 3.0 + vRandom * 0.87) * 0.5 + 0.5);


    vec2 dirNC = normalize(currentP - prevP);
    vec2 dirPC = normalize(nextP - currentP);
    vec2 dir1 = normalize(currentP - prevP);
    vec2 dir2 = normalize(nextP - currentP);
    vec2 dirF = normalize(dir1 + dir2);
    vec2 dirM = mix(dirPC, dirNC, when_eq(nextP, currentP));
    vec2 dir = mix(dirF, dirM, clamp(when_eq(nextP, currentP) + when_eq(prevP, currentP), 0.0, 1.0));
    vec2 normal = vec2(-dir.y, dir.x);
    normal.x /= aspect;

    float lW = mix(-5.0, 5.0, uTransition);
    float direction = mix(-1.0, 1.0, uSide);
    float ppp = (position.z - position.y) * direction;
    float ww = smoothstep(lW - 1.5, lW - 0.5, ppp) * maxWidth - smoothstep(lW - 0.5, lW + 1.5, ppp) * maxWidth * 2.0;
    float lTransform = mix(1.0, 0.0, uSide);
    w += ww * w * lTransform;
    vOpa = 1.0 - smoothstep(lW + 0.5, lW + 1.5, ppp);

    normal *= 0.5 * w;
    finalPosition.xy += normal * side;
    gl_Position = finalPosition;
}

#!SHADER: Fragment

#require(rgb2hsv.fs)

void main() {

    vec2 uvMask = vUv;
    uvMask.x += uRoll;
    uvMask.x += time * 0.1;
    vec3 mask = texture2D(tMask, uvMask).rgb;

    #require(CarColor.fs)

    gl_FragColor.rgb = mix(color, uDisabledColor, uDisable);

    gl_FragColor.a = max(max(mask.r, mask.g), mask.b);

    // fade out every now and again
    gl_FragColor.a *= smoothstep(-0.8, -0.6, sin(vUv.x * 1.0 - time * 0.2 - uAddedTime - uRoll * 0.9 + vRandom * 4.32));
    gl_FragColor.a *= smoothstep(-0.9, -0.7, sin(vUv.x * 1.0 - time * 0.4 - uAddedTime - uRoll * 1.2 + vRandom * 2.12));

    // gl_FragColor.a *= smoothstep(-0.8, -0.6, sin(vUv.x * 1.0 - uRoll * 0.9 + vRandom * 4.32));
    // gl_FragColor.a *= smoothstep(-0.9, -0.7, sin(vUv.x * 1.0 - uRoll * 1.2 + vRandom * 2.12));

    gl_FragColor.a *= uOpacity * vOpa;
}
{@}CarFlowSurroundings.glsl{@}#!ATTRIBUTES

attribute vec3 previous;
attribute vec3 next;
attribute float side;
attribute float width;
attribute float indexNum;
attribute vec2 uv2;
attribute vec3 offset;

#!UNIFORMS

uniform float uOpacity;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec3 uColorIsolate;
uniform float uDisable;
uniform vec3 uDisabledColor;

uniform sampler2D tMask;
uniform float uRoll;
uniform float uTurn;
uniform float uSpeed;

#!VARYINGS
varying float vLineIndex;
varying vec2 vUv;
varying vec2 vUv2;
varying float vRandom;
varying vec3 vPos;

#!SHADER: Vertex

#require(range.glsl)
#require(simplenoise.glsl)

vec2 when_eq(vec2 x, vec2 y) {
    return 1.0 - abs(sign(x - y));
}

float when_lt(float x, float y) {
    return max(sign(y - x), 0.0);
}

vec2 fix(vec4 i, float aspect) {
    vec2 res = i.xy / i.w;
    res.x *= aspect;
    return res;
}

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

void randomCurveStarts(inout vec3 pos, float r) {
    float strength = smoothstep(1.5, 30.0, pos.z * pos.z);
    // pos.x += uTurn * (z * z * 0.05) * strength - uTurn * 0.25 * strength;
    float rand = sin((r * 0.5 + 0.5) * time * 0.2 + uRoll * 0.8 + r * 3.14) * (r * 0.5 + 0.5) * 0.3;
    pos.y += rand * strength * 1.2;
    pos.x += rand * strength * 0.5;
}

void turnCurves(inout vec3 pos) {
    float z = pos.z;
    float strFront = smoothstep(-1.0, 10.0, pos.z);
    pos.x += uTurn * (z * z * 0.05) * strFront;
    float strBack = smoothstep(-1.0, -10.0, pos.z);
    pos.x += uTurn * (z * z * 0.05) * strBack;
}

const float maxWidth = 2.5;

void main() {
    float aspect = resolution.x / resolution.y;

    vUv = uv;
    vUv2 = uv2;
    vLineIndex = indexNum;
    vRandom = random(vec2(indexNum + 2.3));

    vec3 pos = position + offset;
    vec3 prevPos = previous + offset;
    vec3 nextPos = next + offset;

    turnCurves(pos);
    turnCurves(prevPos);
    turnCurves(nextPos);

    vPos = (modelViewMatrix * vec4(pos, 1.0)).xyz;

    mat4 m = projectionMatrix * modelViewMatrix;
    vec4 finalPosition = m * vec4(pos, 1.0);
    vec4 pPos = m * vec4(prevPos, 1.0);
    vec4 nPos = m * vec4(nextPos, 1.0);
    vec2 currentP = fix(finalPosition, aspect);
    vec2 prevP = fix(pPos, aspect);
    vec2 nextP = fix(nPos, aspect);

    float w = 0.1 * width;

    w *= 1.0 - 0.5 * (sin(vUv2.x * 1.3 * 1.0 - time * 2.93 - uRoll * 1.0 + vRandom * 2.31) * 0.5 + 0.5);
    w *= 1.0 - 0.3 * (sin(vUv2.x * 1.3 * 3.0 - time * 1.34 - uRoll * 1.0 + vRandom * 0.87) * 0.5 + 0.5);

    vec2 dirNC = normalize(currentP - prevP);
    vec2 dirPC = normalize(nextP - currentP);
    vec2 dir1 = normalize(currentP - prevP);
    vec2 dir2 = normalize(nextP - currentP);
    vec2 dirF = normalize(dir1 + dir2);
    vec2 dirM = mix(dirPC, dirNC, when_eq(nextP, currentP));
    vec2 dir = mix(dirF, dirM, clamp(when_eq(nextP, currentP) + when_eq(prevP, currentP), 0.0, 1.0));
    vec2 normal = vec2(-dir.y, dir.x);
    normal.x /= aspect;

    normal *= 0.5 * w;
    finalPosition.xy += normal * side;
    gl_Position = finalPosition;
}

#!SHADER: Fragment

#require(rgb2hsv.fs)
#require(range.glsl)
#require(eases.glsl)


void main() {
    vec2 uvMask = vUv;
    uvMask.x += uRoll;
    uvMask.x += time * 0.1;
    vec3 mask = texture2D(tMask, uvMask).rgb;

    #require(CarColor.fs)

    gl_FragColor.rgb = mix(color, uDisabledColor, uDisable);
    gl_FragColor.a = max(max(mask.r, mask.g), mask.b);

    float speedd = uSpeed * 0.6;
    float minLimit = 0.15 * step(speedd, 0.0); // this makes the lines disappear faster when driving backwards, which is nicer
    float speed = 1.0 - quarticIn(crange(abs(speedd), 0.05 + minLimit, 0.98, 0.0, 1.0)); //car speed
    gl_FragColor.a *= smoothstep(-0.8 + 1.8 * speed, -0.5 + 1.5 * speed, sin(vUv2.x * 0.5 - time * 1.0 - uRoll * 2.0 + vRandom * 4.32));
    gl_FragColor.a *= smoothstep(0.5 + 0.5 * speed, 0.3 + 0.7 * speed, sin(vUv2.x * 0.7 - time * 0.6 * vRandom - uRoll * 1.0 + vRandom * 2.91));

    // fade close to the camera
    gl_FragColor.a *= smoothstep(1.0, 3.0, length(vPos));

    gl_FragColor.a *= uOpacity;
}
{@}CarForm.glsl{@}#!ATTRIBUTES

attribute vec3 previous;
attribute vec3 next;
attribute float side;
attribute float width;
attribute float lineIndex;
attribute vec2 uv2;

#!UNIFORMS

uniform float uOpacity;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec3 uColorIsolate;
uniform float uTransition;
uniform float uSide;
uniform float uDisable;
uniform float uAddedTime;
uniform vec3 uDisabledColor;

uniform sampler2D tMask;
uniform float uRoll;
uniform float uTurn;

#!VARYINGS
varying float vLineIndex;
varying vec2 vUv;
varying vec2 vUv2;
varying float vRandom;
varying float vOpa;


#!SHADER: Vertex

#require(range.glsl)
#require(simplenoise.glsl)

vec2 when_eq(vec2 x, vec2 y) {
    return 1.0 - abs(sign(x - y));
}

float when_lt(float x, float y) {
    return max(sign(y - x), 0.0);
}

vec2 fix(vec4 i, float aspect) {
    vec2 res = i.xy / i.w;
    res.x *= aspect;
    return res;
}

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

void turnCurves(inout vec3 pos) {
    // vec2 offset = vec2(0.82, 1.27);
    // offset.x *= mix(1.0, -1.0, when_lt(pos.x, 0.0));
    // float turn = uTurn * 0.5;
    // turn *= when_lt(-pos.z, 0.0);

    // pos.xz -= offset;
    // rotate2D(pos.xz, turn);
    // pos.xz += offset;

    float z = pos.z;
    pos.x += uTurn * (z * z * 0.15) - uTurn * 0.25;
}

const float maxWidth = 2.5;

void main() {
    float aspect = resolution.x / resolution.y;

    vUv = uv;
    vUv2 = uv2;
    vLineIndex = lineIndex;
    vRandom = random(vec2(lineIndex));

    vec3 pos = position;
    vec3 prevPos = previous;
    vec3 nextPos = next;

    // Turn curves to face direction
    // turnCurves(pos);
    // turnCurves(prevPos);
    // turnCurves(nextPos);

    mat4 m = projectionMatrix * modelViewMatrix;
    vec4 finalPosition = m * vec4(pos, 1.0);
    vec4 pPos = m * vec4(prevPos, 1.0);
    vec4 nPos = m * vec4(nextPos, 1.0);
    vec2 currentP = fix(finalPosition, aspect);
    vec2 prevP = fix(pPos, aspect);
    vec2 nextP = fix(nPos, aspect);

    float w = 0.2 * width;

    w *= 1.0 - 0.5 * (sin(vUv2.x * 2. * 1.0 - time * 2.93 - uRoll * 1.0 + vRandom * 2.31) * 0.5 + 0.5);
    w *= 1.0 - 0.3 * (sin(vUv2.x * 2. * 3.0 - time * 1.34 - uRoll * 1.0 + vRandom * 0.87) * 0.5 + 0.5);


    vec2 dirNC = normalize(currentP - prevP);
    vec2 dirPC = normalize(nextP - currentP);
    vec2 dir1 = normalize(currentP - prevP);
    vec2 dir2 = normalize(nextP - currentP);
    vec2 dirF = normalize(dir1 + dir2);
    vec2 dirM = mix(dirPC, dirNC, when_eq(nextP, currentP));
    vec2 dir = mix(dirF, dirM, clamp(when_eq(nextP, currentP) + when_eq(prevP, currentP), 0.0, 1.0));
    vec2 normal = vec2(-dir.y, dir.x);
    normal.x /= aspect;

    float lW = mix(-5.0, 5.0, uTransition);
    float direction = mix(-1.0, 1.0, uSide);
    float ppp = (position.z - position.y) * direction;
    float ww = smoothstep(lW - 1.5, lW - 0.5, ppp) * maxWidth - smoothstep(lW - 0.5, lW + 1.5, ppp) * maxWidth * 2.0;
    float lTransform = mix(1.0, 0.0, uSide);
    w += ww * w * lTransform;
    vOpa = 1.0 - smoothstep(lW + 0.5, lW + 1.5, ppp);

    normal *= 0.5 * w;
    finalPosition.xy += normal * side;
    gl_Position = finalPosition;
}

#!SHADER: Fragment

#require(rgb2hsv.fs)

void main() {

    vec2 uvMask = vUv;
    uvMask.x += uRoll;
    uvMask.x += time * 0.1;
    vec3 mask = texture2D(tMask, uvMask).rgb;

    #require(CarColor.fs)

    gl_FragColor.rgb = mix(color, uDisabledColor, uDisable);

    gl_FragColor.a = max(max(mask.r, mask.g), mask.b);

    // fade out every now and again
    gl_FragColor.a *= smoothstep(-0.9, -0.7, sin(vUv2.x * 1.0 - time * 1.0 - uAddedTime - uRoll * 0.3 + vRandom * 4.32));

    gl_FragColor.a *= uOpacity * vOpa;
}
{@}CarGroundPattern.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMap;
uniform float uScale;
uniform float uDriving;
uniform float uRotation;
uniform float uCarRoll;
uniform float uAlpha;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

void main() {
    vec3 color = vec3(0.0);

    float dist = length(vUv - 0.5);
    float disc = smoothstep(0.07, 0.02, abs(dist - 0.27 * (1.0 - uDriving * 0.2)) );
    float sides = smoothstep(0.03, 0.1, abs(vUv.x - 0.5)) * smoothstep(0.5, 0.0, abs(vUv.y - 0.5));
    disc *= sides;

    vec2 uvDots = vUv * uScale;
    uvDots.y -= uCarRoll * uScale * 0.2;

    float dots = texture2D(tMap, uvDots).g;
    disc *= dots;

    color += disc * 0.8;
    color *= 1.0 - uDriving;

    gl_FragColor.rgb = color;
    gl_FragColor.a = uAlpha;
}
{@}CarMask.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    gl_FragColor.rgb = vec3(vUv, 1.0);
    gl_FragColor.a = 1.0;
}
{@}CarQuad.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMap;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    gl_FragColor = texture2D(tMap, vUv);
}
{@}CarReflection.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMap;
uniform float uMapOpacity;
uniform float uScale;
uniform float uSpeed;

uniform sampler2D tEnv;
uniform float uEnvOpacity;
uniform vec3 uColor;
uniform float uCarSpeed;
uniform float uCarRoll;
uniform float uGlobalOpacity;
uniform float uTransition;
uniform float uSide;

#!VARYINGS

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vModelNormal;
varying vec3 vPos;
varying vec4 vMPos;
varying float vOpa;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vNormal = normalize(normalMatrix * normal);
    vModelNormal = normalize(normal);
    vMPos = modelMatrix * vec4(position, 1.0);

    float lW = mix(-5.0, 5.0, uTransition);
    float direction = mix(-1.0, 1.0, uSide);
    float ppp = (position.z - position.y) * direction;
    vOpa = 1.0 - smoothstep(lW - 1.5, lW + 1.5, ppp);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(envmap.glsl)
#require(range.glsl)
#require(simplenoise.glsl)

void main() {
    vec3 normal = normalize(vNormal);
    vec3 mNormal = normalize(vModelNormal);

    vec4 vvmpos = vMPos;

    vec3 env = envmap(vvmpos, normal, tEnv).rgb;
    env *= smoothstep(0.2, 0.7, dot(vec3(0.0, 1.0, 0.0), mNormal));
    env = pow(env.rgb, vec3(2.0));
    env *= uEnvOpacity;
    env = mix(env, uColor, env.r);

    gl_FragColor.rgb = env;
    gl_FragColor.a = uGlobalOpacity * vOpa;
}
{@}CarRotate.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMap;
uniform float uScale;
uniform float uRotateSpeed;
uniform float uFadeDist;
uniform float uAlpha;

#!VARYINGS

varying vec2 vUv;
varying vec4 vCenter;
varying vec4 vPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vCenter = viewMatrix * vec4(0.0, 0.0, 0.0, 1.0);
    vPos = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * vPos;
}

#!SHADER: Fragment

#require(transformUV.glsl)

void main() {

    vec2 uv = rotateUV(vUv, 4.0 + time * uRotateSpeed);
    uv = scaleUV(uv, vec2(uScale));
    vec4 arrow = vec4(texture2D(tMap, uv).r) * 1.5;

    float distCenter = length(vCenter);
    arrow *= smoothstep(distCenter + uFadeDist, distCenter, length(vPos.xyz));
    arrow *= smoothstep(0.32, 0.4, length(vUv - 0.5));
    arrow *= uAlpha * 0.25;

    gl_FragColor = arrow;
}
{@}CarWheels.glsl{@}#!ATTRIBUTES

attribute vec3 previous;
attribute vec3 next;
attribute float side;
attribute float width;
attribute float lineIndex;
attribute vec2 uv2;

#!UNIFORMS

uniform float uOpacity;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec3 uColorIsolate;
uniform float uTransition;
uniform float uSide;
uniform float uDisable;
uniform vec3 uDisabledColor;
uniform float uAddedTime;

uniform sampler2D tMask;
uniform float uRoll;
uniform float uTurn;

#!VARYINGS
varying float vLineIndex;
varying vec2 vUv;
varying vec2 vUv2;
varying float vDist;
varying float vRandom;
varying float vOpa;


#!SHADER: Vertex

#require(range.glsl)
#require(simplenoise.glsl)

vec2 when_eq(vec2 x, vec2 y) {
    return 1.0 - abs(sign(x - y));
}

float when_lt(float x, float y) {
    return max(sign(y - x), 0.0);
}

vec2 fix(vec4 i, float aspect) {
    vec2 res = i.xy / i.w;
    res.x *= aspect;
    return res;
}

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

void rotateFrontWheel(inout vec3 pos) {
    vec2 offset = vec2(0.82, 1.27);
    offset.x *= mix(1.0, -1.0, when_lt(pos.x, 0.0));
    float turn = uTurn * 0.5;
    turn *= when_lt(-pos.z, 0.0);

    pos.xz -= offset;
    rotate2D(pos.xz, turn);
    pos.xz += offset;
}

const float maxWidth = 2.5;

void main() {
    float aspect = resolution.x / resolution.y;

    vUv = uv;
    vUv2 = uv2;
    vLineIndex = lineIndex;
    vRandom = random(vec2(lineIndex));

    vec3 pos = position;
    vec3 prevPos = previous;
    vec3 nextPos = next;

    // Rotate front wheels
    rotateFrontWheel(pos);
    rotateFrontWheel(prevPos);
    rotateFrontWheel(nextPos);

    mat4 m = projectionMatrix * modelViewMatrix;
    vec4 finalPosition = m * vec4(pos, 1.0);
    vec4 pPos = m * vec4(prevPos, 1.0);
    vec4 nPos = m * vec4(nextPos, 1.0);
    vec2 currentP = fix(finalPosition, aspect);
    vec2 prevP = fix(pPos, aspect);
    vec2 nextP = fix(nPos, aspect);

    float w = 0.2;

    w *= 1.0 - 0.5 * (sin(vUv.x * 6.28 * 1.0 + time * 2.93 + uRoll * 3.0 + vRandom * 2.31) * 0.5 + 0.5);
    w *= 1.0 - 0.3 * (sin(vUv.x * 6.28 * 3.0 + time * 1.34 + uRoll * 3.0 + vRandom * 0.87) * 0.5 + 0.5);

    vec2 dirNC = normalize(currentP - prevP);
    vec2 dirPC = normalize(nextP - currentP);
    vec2 dir1 = normalize(currentP - prevP);
    vec2 dir2 = normalize(nextP - currentP);
    vec2 dirF = normalize(dir1 + dir2);
    vec2 dirM = mix(dirPC, dirNC, when_eq(nextP, currentP));
    vec2 dir = mix(dirF, dirM, clamp(when_eq(nextP, currentP) + when_eq(prevP, currentP), 0.0, 1.0));
    vec2 normal = vec2(-dir.y, dir.x);
    normal.x /= aspect;

    float lW = mix(-5.0, 5.0, uTransition);
    float direction = mix(-1.0, 1.0, uSide);
    float ppp = (position.z - position.y) * direction;
    float ww = smoothstep(lW - 1.5, lW - 0.5, ppp) * maxWidth - smoothstep(lW - 0.5, lW + 1.5, ppp) * maxWidth * 2.0;
    float lTransform = mix(1.0, 0.0, uSide);
    w += ww * w * lTransform;
    vOpa = 1.0 - smoothstep(lW + 0.5, lW + 1.5, ppp);

    normal *= 0.5 * w;
    vDist = finalPosition.z / 10.0;
    finalPosition.xy += normal * side;
    gl_Position = finalPosition;
}

#!SHADER: Fragment

#require(rgb2hsv.fs)

void main() {

    vec2 uvMask = vUv;
    uvMask.x += uRoll;
    uvMask.x += time * 0.1;
    vec3 mask = texture2D(tMask, uvMask).rgb;

    // vec3 color = uColor1;
    vec3 color = vec3(0.5);

    // Single color isolations
    color = mix(color, uColor1, uColorIsolate.x);
    color = mix(color, uColor2, uColorIsolate.y);
    color = mix(color, uColor3, uColorIsolate.z);

    float colorShine = sin(vUv.x * 6.28 + time * 0.4 + + uAddedTime + uRoll * 4.0 + vRandom * 6.41);
    color = mix(color, vec3(1.0), colorShine);

    gl_FragColor.rgb = mix(color, uDisabledColor, uDisable);

    gl_FragColor.a = max(max(mask.r, mask.g), mask.b);

    gl_FragColor.a *= uOpacity * vOpa;
}
{@}CircleNav.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform float uSize;
uniform float uDPR;
uniform vec3 uColor;
uniform float uRotation;
uniform float uMaskLeft;
uniform float uMaskRight;
uniform float uAlpha;
uniform float uHaloAlpha;
uniform float uHaloAlphaFade;
uniform float uHaloAlphaFade2;
uniform float uDriving;
uniform float uDrivingSpeed;
uniform float uDrivingTurn;
uniform float uGlowStrength;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = vec4(position * 2.0, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(rgb2hsv.fs)
#require(transformUV.glsl)

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

void main() {
    vec2 uv = vUv;

    vec2 center = uv - 0.5;
    center.x *= resolution.x / resolution.y;

    vec2 pixelCenter = gl_FragCoord.xy - (0.5 * resolution);
    pixelCenter /= uDPR;

    // Main colored ring
    float thick = 2.1;
    float ring = smoothstep(uSize - thick, uSize - thick + 2.0, length(pixelCenter)) * smoothstep(uSize + thick, uSize + thick - 2.0, length(pixelCenter));

    rotate2D(center, uRotation);
    rotate2D(center, uDrivingTurn);

    // Arc areas for line
    float angle = atan(-center.y, center.x);
    float rightArc = smoothstep((0.9-0.45*uDrivingSpeed+0.1*sin(time*0.5)) * uHaloAlphaFade2 + 0.000001, (0.1+0.1*uDrivingSpeed-0.1*sin(time*0.3)) * uHaloAlphaFade2, abs(angle));
    float maskAngle = mix(1.0, -1.0, uMaskRight);
    float rightMask = smoothstep(maskAngle, maskAngle + 0.01, angle * uHaloAlphaFade2);

    angle = atan(center.y, -center.x);
    maskAngle = mix(1.0, -1.0, uMaskLeft);
    float leftMask = smoothstep(maskAngle, maskAngle + 0.01, angle * uHaloAlphaFade2);
    float leftArc = smoothstep((0.9-0.45*uDrivingSpeed+0.1*sin(time*0.5)) * uHaloAlphaFade2 + 0.000001, (0.1+0.1*uDrivingSpeed-0.1*sin(time*0.3)) * uHaloAlphaFade2, abs(angle));

    // Faded part
    float halo = smoothstep(uSize, uSize + 2.0, length(pixelCenter));

    float glowOuter = (0.8-uDrivingSpeed*0.2) + uGlowStrength*0.04;
    float glowInner = (0.4+uDrivingSpeed*0.2);

    vec2 offsetHalo = vec2(uSize * glowInner, 0.0);
    rotate2D(offsetHalo, -uDrivingTurn);
    halo *= max(smoothstep(uSize * glowOuter, 0.0, length(pixelCenter + offsetHalo)), smoothstep(uSize * glowOuter, 0.0, length(pixelCenter - offsetHalo)));


    halo *= 0.5+uGlowStrength;

    // Trim lines
    rightArc *= rightMask;
    leftArc *= leftMask;

    ring *= max(leftArc, rightArc) * uHaloAlphaFade2;

    float alpha = ring + (halo * uHaloAlpha * uHaloAlphaFade);

    vec3 color = rgb2hsv(uColor);
    color.x += cnoise(vUv*0.5+time*0.1)*0.03;
    color = hsv2rgb(color);
    color = mix(color, vec3(1.0), ring*(1.0-uDrivingSpeed*0.5));

    gl_FragColor.rgb = color;
    gl_FragColor.a = alpha * uAlpha;
}
{@}CustomiseComposite.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform float uDPR;
uniform sampler2D tProduct;
uniform sampler2D tCrossSection;
uniform float uTransition;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = vec4(position * 2.0, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)

void addProduct(inout vec3 color) {
    float screenWidth = resolution.x / uDPR;
    float screenHeight = resolution.y / uDPR;
    float scale1 = min(1.0, (screenWidth - 90.0) / 1200.0);
    float scale2 = min(1.0, (screenHeight - 30.0) / 745.0);
    float scale = min(scale1, scale2);

    float w = 400.0 * scale;
    float h = 220.0 * scale;

    vec2 uv = vUv - 0.5;
    uv.x *= screenWidth / w;
    uv.y *= screenHeight / h;
    uv.y -= 1.0 * (mix(0.0, 1.0, uTransition));

    // Scale up a bit
    uv /= 1.5;
    uv /= mix(2.0, 1.0, uTransition);

    float square = step(abs(uv.x), 0.5) * step(abs(uv.y), 0.5);

    uv += 0.5;

    vec4 tex = texture2D(tProduct, uv);
    vec3 blend = color * tex.rgb * tex.a + color * (1.0 - tex.a);
    // blend *= 0.5;
    color = mix(color, blend, square);
}

void addCrossSection(inout vec3 color) {
    float screenWidth = resolution.x / uDPR;
    float screenHeight = resolution.y / uDPR;
    float scale1 = min(1.0, (screenWidth - 90.0) / 1200.0);
    float scale2 = min(1.0, (screenHeight - 30.0) / 745.0);
    float scale = min(scale1, scale2);

    float w = 400.0 * scale;
    float h = 220.0 * scale;

    vec2 uv = vUv - 0.5;
    uv.x *= screenWidth / w;
    uv.y *= screenHeight / h;
    uv.y -= 1.0;
    uv.x -= 1.0;

    // Scale up a bit
    uv /= 1.5;

    float square = step(abs(uv.x), 0.5) * step(abs(uv.y), 0.5);

    uv += 0.5;

    vec4 tex = texture2D(tCrossSection, uv);
    vec3 blend = color * tex.rgb * tex.a + color * (1.0 - tex.a);
    color = mix(color, blend, square);
}

void main() {
    vec3 color = vec3(0.98);
    color -= pow(abs(vUv.x - 0.4), 2.0) * 0.4;

    // diffuse to remove banding
    color += random(vUv) * 0.01;

    addProduct(color);
    addCrossSection(color);

    gl_FragColor.rgb = color;
    gl_FragColor.a = 1.0;
}
{@}ProductSoloProductShader.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMatCap;
uniform sampler2D tAO;
uniform sampler2D tEnv;
uniform vec3 uColor1;
//uniform vec3 uColor2;

uniform sampler2D tVariables;
uniform float uVariablesFade;
uniform float uVariablesIndex;

#!VARYINGS

varying vec2 vUv;
varying vec2 vUvMatcap;
varying vec3 vPos;
varying vec4 vMPos;
varying vec3 vNormal;

#!SHADER: Vertex

#require(matcap.vs)

void main() {
    vUv = uv;
    vPos = position;
    vNormal = normalize(normalMatrix * normal);
    vMPos = modelMatrix * vec4(position, 1.0);
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vUvMatcap = reflectMatcap(mvPos, vNormal);
    gl_Position = projectionMatrix * mvPos;
}

#!SHADER: Fragment

#require(rgb2hsv.fs)
#require(envmap.glsl)

void addVariable(inout vec3 color) {
    // uVariablesIndex
    // 0, 1, 2 = coverage - low, average, high
    // 3, 4, 5 = packaging-space - low, average, high
    // 6 = top-face
    // 7, 8, 9 = pass-through - poor, average, best
    float topFace = smoothstep(0.2, 0.1, abs(uVariablesIndex - 6.0));
    float coverage = smoothstep(2.3, 2.2, uVariablesIndex);
    float coverageRange = (1.0 - 0.7 * min(1.0, uVariablesIndex / 2.0)) * 0.8;
    float packaging = smoothstep(2.7, 2.8, uVariablesIndex) * smoothstep(5.3, 5.2, uVariablesIndex);
    float packagingRange = max(0.0, min(1.0, (uVariablesIndex - 3.0) / 2.0)) * 0.1;
    float passThrough = smoothstep(6.7, 6.8, uVariablesIndex);
    float passThroughRange = max(0.0, min(1.0, (uVariablesIndex - 7.0) / 2.0)) * 0.2;

    vec2 uvVariables = vUv;
    uvVariables *= mix(1.0, 5.0, topFace);


    vec3 variablesTex = texture2D(tVariables, uvVariables).rgb;

    // Coverage
    vec3 rgbCoverage = mix(color, mix(color, vec3(1.0), 0.7), smoothstep(coverageRange - 0.01, coverageRange, variablesTex.g));
    variablesTex = mix(variablesTex, rgbCoverage, coverage);

    // Packaging Space
    vec3 rgbPackaging = hsv2rgb(vec3( min(0.7, max(0.0, (1.0 - variablesTex.g * 1.2) * 0.75 + packagingRange)), 1.0, 1.0));
    variablesTex = mix(variablesTex, rgbPackaging, packaging);

    // Top face
    variablesTex = mix(variablesTex, color + variablesTex.g, topFace);

    // Pass through
    vec3 rgbPass = mix(color, hsv2rgb(vec3(1.0 + passThroughRange * 2.5, 1.0, 1.0)), smoothstep(0.8 - passThroughRange, 0.7 - passThroughRange, variablesTex.g));
    variablesTex = mix(variablesTex, rgbPass, passThrough);


    color = mix(color, variablesTex, uVariablesFade);
}

void main() {
    vec3 normal = normalize(vNormal);
    float matcap = texture2D(tMatCap, vUvMatcap).g;
    float ao = texture2D(tAO, vUv).g;
    float env = envmap(vMPos, normal, tEnv).g;

    float tone = mix(1.0, matcap, 0.5);
    tone *= ao * 1.5;

    vec3 color1 = rgb2hsv(uColor1);
    color1.x += smoothstep(0.1, 0.2, color1.x) * 0.15;
    color1.y *= 1.5;
    color1.z *= 0.4;
    color1 = hsv2rgb(color1);

    vec3 color2 = rgb2hsv(uColor1);
    color2.y *= 0.9;
    color2 = hsv2rgb(color2);

    vec3 color = mix(color1, color2, tone);

    float refl = env * env;
    color += refl * 0.02;



    addVariable(color);



    gl_FragColor.rgb = color;
    gl_FragColor.a = 1.0;
}
{@}CarEngine.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMatCap;
uniform sampler2D tAO;
uniform sampler2D tEnv;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec3 uColor4;
uniform vec3 uColor5;
uniform float uVisible;

#!VARYINGS

varying vec2 vUv;
varying vec2 vUvMatcap;
varying vec3 vPos;
varying vec4 vMPos;
varying vec3 vNormal;

#!SHADER: Vertex

#require(matcap.vs)

void main() {
    vUv = uv;
    vPos = position;
    vNormal = normalize(normalMatrix * normal);
    vMPos = modelMatrix * vec4(position, 1.0);
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vUvMatcap = reflectMatcap(mvPos, vNormal);
    gl_Position = projectionMatrix * mvPos;
}

#!SHADER: Fragment

#require(rgb2hsv.fs)
#require(envmap.glsl)

void main() {

    // Transition alpha clip
    float fadeAlpha = mix(0.0, 1.3, uVisible);
    float fade = smoothstep(fadeAlpha + 0.2, fadeAlpha - 0.2, vPos.y );
    float circles = length(mod(vPos * 30.0, vec3(1.0)) - 0.5);
    if (circles > fade) discard;

    vec3 normal = normalize(vNormal);
    float matcap = texture2D(tMatCap, vUvMatcap).g;
    float ao = texture2D(tAO, vUv).g;
    float env = envmap(vMPos, normal, tEnv).g;

    float tone = mix(1.0, matcap, 0.5);
    tone *= ao * 1.3;
    vec3 color = mix(uColor2, uColor3, matcap*0.5);
    color = mix(color, uColor4, ao*0.5);
    color = mix(color, uColor1, smoothstep(0.5, 1.0, tone));



    float refl = env * env;
    color += refl * 0.1;

    gl_FragColor.rgb = color;
    gl_FragColor.a = 1.0;
}
{@}GLSectionChoice.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tBg;
uniform float uDPR;
uniform vec2 uOffset;
uniform float uSize;
uniform vec3 uColor;
uniform vec3 uCircleColor;
uniform float uScale;

uniform float uAlpha;
uniform float uBg;
uniform float uCircle;
uniform float uIcon;
uniform float uGlow;
uniform float uFade;
uniform float uDriving;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;

    vec3 pos = position;

    pos.xy /= resolution / uDPR;
    pos.xy *= uSize * 2.0 * uScale;
    pos.xy += uOffset / resolution * uDPR * 2.0; 

    gl_Position = vec4(pos, 1.0);
}

#!SHADER: Fragment

vec4 getCircle() {
    float dist = length(vUv - 0.5) * 2.0;

    float t = 0.12;
    float a = 0.09;
    float alpha = smoothstep(0.5 - a - t, 0.5 - t, dist) * smoothstep(0.5, 0.5 - a, dist);
    vec3 color = uCircleColor;

    alpha *= uCircle * uFade;

    return vec4(color, alpha);
}

vec4 getBG() {
    float dist = length(vUv - 0.5) * 2.0;
    vec2 uv = gl_FragCoord.xy / resolution;
    vec3 color = texture2D(tBg, uv).rgb;
    float alpha = smoothstep(0.47, 0.44, dist);

    return vec4(color, alpha);
}

vec4 getGlow() {
    float dist = length(vUv - 0.5);

    vec3 color = uColor;
    float alpha = pow( max(0.0, 1.0 - ((dist - 0.25) * 4.0)), 2.0) * 0.3;
    alpha *= uGlow * uFade;
    return vec4(color, alpha);
}

void main() {
    vec3 color = vec3(0.0);
    float alpha = 0.0;

    vec4 glow = getGlow();
    color = glow.rgb;
    alpha += glow.a;

    vec4 bg = getBG();
    color = mix(color, bg.rgb, vec3(bg.a));
    alpha += bg.a * uBg;

    vec4 circle = getCircle();
    color = mix(color, circle.rgb, vec3(circle.a));
    alpha += circle.a;

    alpha *= uAlpha * uDriving;

    gl_FragColor.rgb = color;
    gl_FragColor.a = alpha;
}
{@}GLSectionTitle.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tBg;
uniform sampler2D tIcon;
uniform float uDPR;
uniform vec2 uOffset;
uniform float uSize;
uniform vec3 uColor;
uniform float uScale;

uniform float uAlpha;
uniform float uBg;
uniform float uCircle;
uniform float uIcon;
uniform float uGlow;
uniform float uFade;
uniform float uDriving;
uniform float uIconFade;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;

    vec3 pos = position;

    pos.xy /= resolution / uDPR;
    pos.xy *= uSize * 2.0 * uScale;
    pos.xy += uOffset / resolution * uDPR * 2.0;

    gl_Position = vec4(pos, 1.0);
}

#!SHADER: Fragment

vec4 getCircle() {
    float dist = length(vUv - 0.5) * 2.0;

    float t = 0.042;
    float a = 0.05;
    float alpha = smoothstep(0.5 - a - t, 0.5 - t, dist) * smoothstep(0.5, 0.5 - a, dist);
    vec3 color = vec3(1.0);

    alpha *= uCircle * uFade;

    return vec4(color, alpha);
}

vec4 getBG() {
    float dist = length(vUv - 0.5) * 2.0;
    vec2 uv = gl_FragCoord.xy / resolution;
    vec3 color = texture2D(tBg, uv).rgb;
    float alpha = smoothstep(0.48, 0.44, dist);

    return vec4(color, alpha);
}

vec4 getGlow() {
    float dist = length(vUv - 0.5);

    vec3 color = uColor;
    float alpha = pow( max(0.0, 1.0 - ((dist - 0.25) * 4.0)), 2.0) * 0.3;
    alpha *= uGlow * uFade;
    return vec4(color, alpha);
}

vec4 getIcon() {
    vec2 uv = vUv;
    uv -= 0.5;
    uv /= 0.28;
    uv += 0.5;
    vec3 color = vec3(1.0);
    float alpha = texture2D(tIcon, uv).a;

    // Stop stretching on edges for mipmapped textures
    alpha = mix(0.0, alpha, max(0.0, min(1.0, uv.x * 10.0)) );
    alpha = mix(alpha, 0.0, max(0.0, min(1.0, (uv.x - 1.0) * 10.0)) );

    alpha *= uIcon * uFade;

    return vec4(color, alpha);
}

void main() {
    vec3 color = vec3(0.0);
    float alpha = 0.0;

    vec4 glow = getGlow();
    color = glow.rgb;
    alpha += glow.a;

    vec4 bg = getBG();
    color = mix(color, bg.rgb, vec3(bg.a));
    alpha += bg.a * uBg;

    vec4 circle = getCircle();
    color = mix(color, circle.rgb, vec3(circle.a));
    alpha += circle.a;

    vec4 icon = getIcon();
    color = mix(color, icon.rgb, vec3(icon.a)*uIconFade);
    alpha += icon.a;

    alpha *= uAlpha * uDriving;

    gl_FragColor.rgb = color;
    gl_FragColor.a = alpha;
}
{@}AntimatterCopy.fs{@}uniform sampler2D tDiffuse;

varying vec2 vUv;

void main() {
    gl_FragColor = texture2D(tDiffuse, vUv);
}{@}AntimatterCopy.vs{@}varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}{@}AntimatterPass.vs{@}void main() {
    gl_Position = vec4(position, 1.0);
}{@}AntimatterPosition.vs{@}uniform sampler2D tPos;

void main() {
    vec4 decodedPos = texture2D(tPos, position.xy);
    vec3 pos = decodedPos.xyz;

    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_PointSize = 0.02 * (1000.0 / length(mvPosition.xyz));
    gl_Position = projectionMatrix * mvPosition;
}{@}AntimatterBasicFrag.fs{@}void main() {
    gl_FragColor = vec4(1.0);
}{@}antimatter.glsl{@}vec3 getData(sampler2D tex, vec2 uv) {
    return texture2D(tex, uv).xyz;
}

vec4 getData4(sampler2D tex, vec2 uv) {
    return texture2D(tex, uv);
}{@}conditionals.glsl{@}vec4 when_eq(vec4 x, vec4 y) {
  return 1.0 - abs(sign(x - y));
}

vec4 when_neq(vec4 x, vec4 y) {
  return abs(sign(x - y));
}

vec4 when_gt(vec4 x, vec4 y) {
  return max(sign(x - y), 0.0);
}

vec4 when_lt(vec4 x, vec4 y) {
  return max(sign(y - x), 0.0);
}

vec4 when_ge(vec4 x, vec4 y) {
  return 1.0 - when_lt(x, y);
}

vec4 when_le(vec4 x, vec4 y) {
  return 1.0 - when_gt(x, y);
}

vec3 when_eq(vec3 x, vec3 y) {
  return 1.0 - abs(sign(x - y));
}

vec3 when_neq(vec3 x, vec3 y) {
  return abs(sign(x - y));
}

vec3 when_gt(vec3 x, vec3 y) {
  return max(sign(x - y), 0.0);
}

vec3 when_lt(vec3 x, vec3 y) {
  return max(sign(y - x), 0.0);
}

vec3 when_ge(vec3 x, vec3 y) {
  return 1.0 - when_lt(x, y);
}

vec3 when_le(vec3 x, vec3 y) {
  return 1.0 - when_gt(x, y);
}

vec2 when_eq(vec2 x, vec2 y) {
  return 1.0 - abs(sign(x - y));
}

vec2 when_neq(vec2 x, vec2 y) {
  return abs(sign(x - y));
}

vec2 when_gt(vec2 x, vec2 y) {
  return max(sign(x - y), 0.0);
}

vec2 when_lt(vec2 x, vec2 y) {
  return max(sign(y - x), 0.0);
}

vec2 when_ge(vec2 x, vec2 y) {
  return 1.0 - when_lt(x, y);
}

vec2 when_le(vec2 x, vec2 y) {
  return 1.0 - when_gt(x, y);
}

float when_eq(float x, float y) {
  return 1.0 - abs(sign(x - y));
}

float when_neq(float x, float y) {
  return abs(sign(x - y));
}

float when_gt(float x, float y) {
  return max(sign(x - y), 0.0);
}

float when_lt(float x, float y) {
  return max(sign(y - x), 0.0);
}

float when_ge(float x, float y) {
  return 1.0 - when_lt(x, y);
}

float when_le(float x, float y) {
  return 1.0 - when_gt(x, y);
}

vec4 and(vec4 a, vec4 b) {
  return a * b;
}

vec4 or(vec4 a, vec4 b) {
  return min(a + b, 1.0);
}

vec4 Not(vec4 a) {
  return 1.0 - a;
}

vec3 and(vec3 a, vec3 b) {
  return a * b;
}

vec3 or(vec3 a, vec3 b) {
  return min(a + b, 1.0);
}

vec3 Not(vec3 a) {
  return 1.0 - a;
}

vec2 and(vec2 a, vec2 b) {
  return a * b;
}

vec2 or(vec2 a, vec2 b) {
  return min(a + b, 1.0);
}


vec2 Not(vec2 a) {
  return 1.0 - a;
}

float and(float a, float b) {
  return a * b;
}

float or(float a, float b) {
  return min(a + b, 1.0);
}

float Not(float a) {
  return 1.0 - a;
}{@}curl.glsl{@}#require(simplex3d.glsl)

vec3 snoiseVec3( vec3 x ){
    
    float s  = snoise(vec3( x ));
    float s1 = snoise(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ));
    float s2 = snoise(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ));
    vec3 c = vec3( s , s1 , s2 );
    return c;
    
}


vec3 curlNoise( vec3 p ){
    
    const float e = 1e-1;
    vec3 dx = vec3( e   , 0.0 , 0.0 );
    vec3 dy = vec3( 0.0 , e   , 0.0 );
    vec3 dz = vec3( 0.0 , 0.0 , e   );
    
    vec3 p_x0 = snoiseVec3( p - dx );
    vec3 p_x1 = snoiseVec3( p + dx );
    vec3 p_y0 = snoiseVec3( p - dy );
    vec3 p_y1 = snoiseVec3( p + dy );
    vec3 p_z0 = snoiseVec3( p - dz );
    vec3 p_z1 = snoiseVec3( p + dz );
    
    float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
    float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
    float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;
    
    const float divisor = 1.0 / ( 2.0 * e );
    return normalize( vec3( x , y , z ) * divisor );
}{@}eases.glsl{@}#ifndef PI
#define PI 3.141592653589793
#endif

#ifndef HALF_PI
#define HALF_PI 1.5707963267948966
#endif

float backInOut(float t) {
  float f = t < 0.5
    ? 2.0 * t
    : 1.0 - (2.0 * t - 1.0);

  float g = pow(f, 3.0) - f * sin(f * PI);

  return t < 0.5
    ? 0.5 * g
    : 0.5 * (1.0 - g) + 0.5;
}

float backIn(float t) {
  return pow(t, 3.0) - t * sin(t * PI);
}

float backOut(float t) {
  float f = 1.0 - t;
  return 1.0 - (pow(f, 3.0) - f * sin(f * PI));
}

float bounceOut(float t) {
  const float a = 4.0 / 11.0;
  const float b = 8.0 / 11.0;
  const float c = 9.0 / 10.0;

  const float ca = 4356.0 / 361.0;
  const float cb = 35442.0 / 1805.0;
  const float cc = 16061.0 / 1805.0;

  float t2 = t * t;

  return t < a
    ? 7.5625 * t2
    : t < b
      ? 9.075 * t2 - 9.9 * t + 3.4
      : t < c
        ? ca * t2 - cb * t + cc
        : 10.8 * t * t - 20.52 * t + 10.72;
}

float bounceIn(float t) {
  return 1.0 - bounceOut(1.0 - t);
}

float bounceInOut(float t) {
  return t < 0.5
    ? 0.5 * (1.0 - bounceOut(1.0 - t * 2.0))
    : 0.5 * bounceOut(t * 2.0 - 1.0) + 0.5;
}

float circularInOut(float t) {
  return t < 0.5
    ? 0.5 * (1.0 - sqrt(1.0 - 4.0 * t * t))
    : 0.5 * (sqrt((3.0 - 2.0 * t) * (2.0 * t - 1.0)) + 1.0);
}

float circularIn(float t) {
  return 1.0 - sqrt(1.0 - t * t);
}

float circularOut(float t) {
  return sqrt((2.0 - t) * t);
}

float cubicInOut(float t) {
  return t < 0.5
    ? 4.0 * t * t * t
    : 0.5 * pow(2.0 * t - 2.0, 3.0) + 1.0;
}

float cubicIn(float t) {
  return t * t * t;
}

float cubicOut(float t) {
  float f = t - 1.0;
  return f * f * f + 1.0;
}

float elasticInOut(float t) {
  return t < 0.5
    ? 0.5 * sin(+13.0 * HALF_PI * 2.0 * t) * pow(2.0, 10.0 * (2.0 * t - 1.0))
    : 0.5 * sin(-13.0 * HALF_PI * ((2.0 * t - 1.0) + 1.0)) * pow(2.0, -10.0 * (2.0 * t - 1.0)) + 1.0;
}

float elasticIn(float t) {
  return sin(13.0 * t * HALF_PI) * pow(2.0, 10.0 * (t - 1.0));
}

float elasticOut(float t) {
  return sin(-13.0 * (t + 1.0) * HALF_PI) * pow(2.0, -10.0 * t) + 1.0;
}

float expoInOut(float t) {
  return t == 0.0 || t == 1.0
    ? t
    : t < 0.5
      ? +0.5 * pow(2.0, (20.0 * t) - 10.0)
      : -0.5 * pow(2.0, 10.0 - (t * 20.0)) + 1.0;
}

float expoIn(float t) {
  return t == 0.0 ? t : pow(2.0, 10.0 * (t - 1.0));
}

float expoOut(float t) {
  return t == 1.0 ? t : 1.0 - pow(2.0, -10.0 * t);
}

float linear(float t) {
  return t;
}

float quadraticInOut(float t) {
  float p = 2.0 * t * t;
  return t < 0.5 ? p : -p + (4.0 * t) - 1.0;
}

float quadraticIn(float t) {
  return t * t;
}

float quadraticOut(float t) {
  return -t * (t - 2.0);
}

float quarticInOut(float t) {
  return t < 0.5
    ? +8.0 * pow(t, 4.0)
    : -8.0 * pow(t - 1.0, 4.0) + 1.0;
}

float quarticIn(float t) {
  return pow(t, 4.0);
}

float quarticOut(float t) {
  return pow(t - 1.0, 3.0) * (1.0 - t) + 1.0;
}

float qinticInOut(float t) {
  return t < 0.5
    ? +16.0 * pow(t, 5.0)
    : -0.5 * pow(2.0 * t - 2.0, 5.0) + 1.0;
}

float qinticIn(float t) {
  return pow(t, 5.0);
}

float qinticOut(float t) {
  return 1.0 - (pow(t - 1.0, 5.0));
}

float sineInOut(float t) {
  return -0.5 * (cos(PI * t) - 1.0);
}

float sineIn(float t) {
  return sin((t - 1.0) * HALF_PI) + 1.0;
}

float sineOut(float t) {
  return sin(t * HALF_PI);
}
{@}ColorMaterial.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform vec3 color;

#!VARYINGS

#!SHADER: ColorMaterial.vs
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: ColorMaterial.fs
void main() {
    gl_FragColor = vec4(color, 1.0);
}{@}DebugCamera.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS
varying vec3 vColor;

#!SHADER: DebugCamera.vs
void main() {
    vColor = mix(vec3(1.0), vec3(1.0, 0.0, 0.0), step(position.z, 0.0));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: DebugCamera.fs
void main() {
    gl_FragColor = vec4(vColor, 1.0);
}{@}ScreenQuad.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;

#!SHADER: ScreenQuad.vs
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: ScreenQuad.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.a = 1.0;
}{@}TestMaterial.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS
varying vec3 vNormal;

#!SHADER: TestMaterial.vs
void main() {
    vNormal = normalMatrix * normal;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: TestMaterial.fs
void main() {
    gl_FragColor = vec4(vNormal, 1.0);
}{@}TextureMaterial.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;

#!SHADER: TextureMaterial.vs
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: TextureMaterial.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.rgb /= gl_FragColor.a;
}{@}BlitPass.fs{@}void main() {
    gl_FragColor = texture2D(tDiffuse, vUv);
    gl_FragColor.a = 1.0;
}{@}NukePass.vs{@}varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}{@}ShadowDepth.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS

#!SHADER: ShadowDepth.vs
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: ShadowDepth.fs
void main() {
    gl_FragColor = vec4(vec3(gl_FragCoord.z), 1.0);
}{@}instance.vs{@}vec3 transformNormal(vec3 n, vec4 orientation) {
    vec3 ncN = cross(orientation.xyz, n);
    n = ncN * (2.0 * orientation.w) + (cross(orientation.xyz, ncN) * 2.0 + n);
    return n;
}

vec3 transformPosition(vec3 position, vec3 offset, vec3 scale, vec4 orientation) {
    vec3 pos = position;
    pos *= scale;

    pos = pos + 2.0 * cross(orientation.xyz, cross(orientation.xyz, pos) + orientation.w * pos);
    pos += offset;
    return pos;
}

vec3 transformPosition(vec3 position, vec3 offset, vec4 orientation) {
    vec3 pos = position;

    pos = pos + 2.0 * cross(orientation.xyz, cross(orientation.xyz, pos) + orientation.w * pos);
    pos += offset;
    return pos;
}

vec3 transformPosition(vec3 position, vec3 offset, float scale, vec4 orientation) {
    return transformPosition(position, offset, vec3(scale), orientation);
}

vec3 transformPosition(vec3 position, vec3 offset) {
    return position + offset;
}

vec3 transformPosition(vec3 position, vec3 offset, float scale) {
    vec3 pos = position * scale;
    return pos + offset;
}

vec3 transformPosition(vec3 position, vec3 offset, vec3 scale) {
    vec3 pos = position * scale;
    return pos + offset;
}{@}lights.fs{@}vec3 worldLight(vec3 pos, vec3 vpos) {
    vec4 mvPos = modelViewMatrix * vec4(vpos, 1.0);
    vec4 worldPosition = viewMatrix * vec4(pos, 1.0);
    return worldPosition.xyz - mvPos.xyz;
}{@}lights.vs{@}vec3 worldLight(vec3 pos) {
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vec4 worldPosition = viewMatrix * vec4(pos, 1.0);
    return worldPosition.xyz - mvPos.xyz;
}

vec3 worldLight(vec3 lightPos, vec3 localPos) {
    vec4 mvPos = modelViewMatrix * vec4(localPos, 1.0);
    vec4 worldPosition = viewMatrix * vec4(lightPos, 1.0);
    return worldPosition.xyz - mvPos.xyz;
}{@}shadows.fs{@}float shadowCompare(sampler2D map, vec2 coords, float compare) {
    return step(compare, texture2D(map, coords).r);
}

float shadowLerp(sampler2D map, vec2 coords, float compare, float size) {
    const vec2 offset = vec2(0.0, 1.0);

    vec2 texelSize = vec2(1.0) / size;
    vec2 centroidUV = floor(coords * size + 0.5) / size;

    float lb = shadowCompare(map, centroidUV + texelSize * offset.xx, compare);
    float lt = shadowCompare(map, centroidUV + texelSize * offset.xy, compare);
    float rb = shadowCompare(map, centroidUV + texelSize * offset.yx, compare);
    float rt = shadowCompare(map, centroidUV + texelSize * offset.yy, compare);

    vec2 f = fract( coords * size + 0.5 );

    float a = mix( lb, lt, f.y );
    float b = mix( rb, rt, f.y );
    float c = mix( a, b, f.x );

    return c;
}

float srange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    float oldRange = oldMax - oldMin;
    float newRange = newMax - newMin;
    return (((oldValue - oldMin) * newRange) / oldRange) + newMin;
}

float shadowrandom(vec3 vin) {
    vec3 v = vin * 0.1;
    float t = v.z * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += srange(sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0) + sin(v.x * -3.5 / s + t * 4.0) + sin(v.x * -2.5 / s + t * 7.1), -1.0, 1.0, -0.3, 0.3);
    noise += srange(sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0) + sin(v.y * 2.6 / s + t * 8.0) + sin(v.y * -2.6 / s + t * 4.5), -1.0, 1.0, -0.3, 0.3);
    return noise;
}

float shadowLookup(sampler2D map, vec3 coords, float size, float compare, vec3 wpos) {
    float shadow = 1.0;

    #if defined(SHADOW_MAPS)
    bvec4 inFrustumVec = bvec4 (coords.x >= 0.0, coords.x <= 1.0, coords.y >= 0.0, coords.y <= 1.0);
    bool inFrustum = all(inFrustumVec);
    bvec2 frustumTestVec = bvec2(inFrustum, coords.z <= 1.0);
    bool frustumTest = all(frustumTestVec);

    if (frustumTest) {
        vec2 texelSize = vec2(1.0) / size;

        float dx0 = -texelSize.x;
        float dy0 = -texelSize.y;
        float dx1 = +texelSize.x;
        float dy1 = +texelSize.y;

        float rnoise = shadowrandom(wpos) * 0.0015;
        dx0 += rnoise;
        dy0 -= rnoise;
        dx1 += rnoise;
        dy1 -= rnoise;

        #if defined(SHADOWS_MED)
        shadow += shadowCompare(map, coords.xy + vec2(0.0, dy0), compare);
//        shadow += shadowCompare(map, coords.xy + vec2(dx1, dy0), compare);
        shadow += shadowCompare(map, coords.xy + vec2(dx0, 0.0), compare);
        shadow += shadowCompare(map, coords.xy, compare);
        shadow += shadowCompare(map, coords.xy + vec2(dx1, 0.0), compare);
//        shadow += shadowCompare(map, coords.xy + vec2(dx0, dy1), compare);
        shadow += shadowCompare(map, coords.xy + vec2(0.0, dy1), compare);
        shadow /= 5.0;

        #elif defined(SHADOWS_HIGH)
        shadow = shadowLerp(map, coords.xy + vec2(dx0, dy0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(0.0, dy0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx1, dy0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx0, 0.0), compare, size);
        shadow += shadowLerp(map, coords.xy, compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx1, 0.0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx0, dy1), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(0.0, dy1), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx1, dy1), compare, size);
        shadow /= 9.0;

        #else
        shadow = shadowCompare(map, coords.xy, compare);
        #endif
    }

    #endif

    return clamp(shadow, 0.0, 1.0);
}

vec3 transformShadowLight(vec3 pos, vec3 vpos) {
    vec4 mvPos = modelViewMatrix * vec4(vpos, 1.0);
    vec4 worldPosition = viewMatrix * vec4(pos, 1.0);
    return normalize(worldPosition.xyz - mvPos.xyz);
}

float getShadow(vec3 pos, vec3 normal, float bias) {
    float shadow = 1.0;
    #if defined(SHADOW_MAPS)

    #pragma unroll_loop
    for (int i = 0; i < SHADOW_COUNT; i++) {
        vec4 shadowMapCoords = shadowMatrix[i] * vec4(pos, 1.0);
        vec3 coords = (shadowMapCoords.xyz / shadowMapCoords.w) * vec3(0.5) + vec3(0.5);

        float lookup = shadowLookup(shadowMap[i], coords, shadowSize[i], coords.z - bias, pos);
        lookup += mix(1.0 - step(0.002, dot(transformShadowLight(shadowLightPos[i], pos), normal)), 0.0, step(999.0, normal.x));
        shadow *= clamp(lookup, 0.0, 1.0);
    }

    #endif
    return shadow;
}

float getShadow(vec3 pos, vec3 normal) {
    return getShadow(pos, normal, 0.0);
}

float getShadow(vec3 pos, float bias) {
    return getShadow(pos, vec3(99999.0), bias);
}

float getShadow(vec3 pos) {
    return getShadow(pos, vec3(99999.0), 0.0);
}{@}envmap.glsl{@}#define RECIPROCAL_PI 0.31830988618
#define RECIPROCAL_PI2 0.15915494

vec3 inverseTformDir(in vec3 dir, in mat4 matrix) {
	return normalize((vec4(dir, 0.0) * matrix).xyz);
}

// For use in fragment shader alone
vec4 envmap(vec4 mPos, vec3 normal, sampler2D uEnv) {

    // Requires uniforms cameraPosition, viewMatrix
    vec3 cameraToVertex = normalize(mPos.xyz - cameraPosition);
    vec3 worldNormal = inverseTformDir(normalize(normal), viewMatrix);
    vec3 reflect = normalize(reflect(cameraToVertex, worldNormal));

    vec2 uv;
    uv.y = asin(clamp(reflect.y, -1.0, 1.0)) * RECIPROCAL_PI + 0.5;
    uv.x = atan(reflect.z, reflect.x) * RECIPROCAL_PI2 + 0.5;
    return texture2D(uEnv, uv);
}

//viewMatrix * mvPos{@}FXAA.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS
varying vec2 v_rgbNW;
varying vec2 v_rgbNE;
varying vec2 v_rgbSW;
varying vec2 v_rgbSE;
varying vec2 v_rgbM;

#!SHADER: FXAA.vs

varying vec2 vUv;

void main() {
    vUv = uv;

    vec2 fragCoord = uv * resolution;
    vec2 inverseVP = 1.0 / resolution.xy;
    v_rgbNW = (fragCoord + vec2(-1.0, -1.0)) * inverseVP;
    v_rgbNE = (fragCoord + vec2(1.0, -1.0)) * inverseVP;
    v_rgbSW = (fragCoord + vec2(-1.0, 1.0)) * inverseVP;
    v_rgbSE = (fragCoord + vec2(1.0, 1.0)) * inverseVP;
    v_rgbM = vec2(fragCoord * inverseVP);

    gl_Position = vec4(position, 1.0);
}

#!SHADER: FXAA.fs

#ifndef FXAA_REDUCE_MIN
    #define FXAA_REDUCE_MIN   (1.0/ 128.0)
#endif
#ifndef FXAA_REDUCE_MUL
    #define FXAA_REDUCE_MUL   (1.0 / 8.0)
#endif
#ifndef FXAA_SPAN_MAX
    #define FXAA_SPAN_MAX     8.0
#endif

vec4 fxaa(sampler2D tex, vec2 fragCoord, vec2 resolution,
            vec2 v_rgbNW, vec2 v_rgbNE,
            vec2 v_rgbSW, vec2 v_rgbSE,
            vec2 v_rgbM) {
    vec4 color;
    mediump vec2 inverseVP = vec2(1.0 / resolution.x, 1.0 / resolution.y);
    vec3 rgbNW = texture2D(tex, v_rgbNW).xyz;
    vec3 rgbNE = texture2D(tex, v_rgbNE).xyz;
    vec3 rgbSW = texture2D(tex, v_rgbSW).xyz;
    vec3 rgbSE = texture2D(tex, v_rgbSE).xyz;
    vec4 texColor = texture2D(tex, v_rgbM);
    vec3 rgbM  = texColor.xyz;
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    mediump vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
                          (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
              max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
              dir * rcpDirMin)) * inverseVP;

    vec3 rgbA = 0.5 * (
        texture2D(tex, fragCoord * inverseVP + dir * (1.0 / 3.0 - 0.5)).xyz +
        texture2D(tex, fragCoord * inverseVP + dir * (2.0 / 3.0 - 0.5)).xyz);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture2D(tex, fragCoord * inverseVP + dir * -0.5).xyz +
        texture2D(tex, fragCoord * inverseVP + dir * 0.5).xyz);

    float lumaB = dot(rgbB, luma);
    if ((lumaB < lumaMin) || (lumaB > lumaMax))
        color = vec4(rgbA, texColor.a);
    else
        color = vec4(rgbB, texColor.a);
    return color;
}

void main() {
    vec2 fragCoord = vUv * resolution;
    gl_FragColor = fxaa(tDiffuse, fragCoord, resolution, v_rgbNW, v_rgbNE, v_rgbSW, v_rgbSE, v_rgbM);
    gl_FragColor.a = 1.0;
}{@}gaussianblur.fs{@}vec4 blur13(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.411764705882353) * direction;
  vec2 off2 = vec2(3.2941176470588234) * direction;
  vec2 off3 = vec2(5.176470588235294) * direction;
  color += texture2D(image, uv) * 0.1964825501511404;
  color += texture2D(image, uv + (off1 / resolution)) * 0.2969069646728344;
  color += texture2D(image, uv - (off1 / resolution)) * 0.2969069646728344;
  color += texture2D(image, uv + (off2 / resolution)) * 0.09447039785044732;
  color += texture2D(image, uv - (off2 / resolution)) * 0.09447039785044732;
  color += texture2D(image, uv + (off3 / resolution)) * 0.010381362401148057;
  color += texture2D(image, uv - (off3 / resolution)) * 0.010381362401148057;
  return color;
}

vec4 blur5(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3333333333333333) * direction;
  color += texture2D(image, uv) * 0.29411764705882354;
  color += texture2D(image, uv + (off1 / resolution)) * 0.35294117647058826;
  color += texture2D(image, uv - (off1 / resolution)) * 0.35294117647058826;
  return color;
}

vec4 blur9(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3846153846) * direction;
  vec2 off2 = vec2(3.2307692308) * direction;
  color += texture2D(image, uv) * 0.2270270270;
  color += texture2D(image, uv + (off1 / resolution)) * 0.3162162162;
  color += texture2D(image, uv - (off1 / resolution)) * 0.3162162162;
  color += texture2D(image, uv + (off2 / resolution)) * 0.0702702703;
  color += texture2D(image, uv - (off2 / resolution)) * 0.0702702703;
  return color;
}{@}glscreenprojection.glsl{@}vec2 frag_coord(vec4 glPos) {
    return ((glPos.xyz / glPos.w) * 0.5 + 0.5).xy;
}

vec2 getProjection(vec3 pos, mat4 projMatrix) {
    vec4 mvpPos = projMatrix * vec4(pos, 1.0);
    return frag_coord(mvpPos);
}

void applyNormal(vec2 pos, mat3 projNormalMatrix) {
    vec3 transformed = projNormalMatrix * vec3(pos, 0.0);
    pos = transformed.xy;
}{@}DefaultText.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS

varying vec2 vUv;

#!SHADER: DefaultText.vs

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: DefaultText.fs

#require(msdf.glsl)

void main() {
    float alpha = msdf(tMap);

    gl_FragColor.rgb = uColor;
    gl_FragColor.a = alpha * uAlpha;
}
{@}msdf.glsl{@}float msdf(sampler2D tMap, vec2 uv) {
    vec3 tex = texture2D(tMap, uv).rgb;
    float signedDist = max(min(tex.r, tex.g), min(max(tex.r, tex.g), tex.b)) - 0.5;

    // TODO: fallback for fwidth for webgl1 (need to enable ext)
    float d = fwidth(signedDist);
    float alpha = smoothstep(-d, d, signedDist);
    if (alpha < 0.01) discard;
    return alpha;
}

float msdf(sampler2D tMap) {
    return msdf(tMap, vUv);
}{@}GLUIObject.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;

#!SHADER: GLUIObject.vs
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: GLUIObject.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.a *= uAlpha;
}{@}GLUIObjectMask.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;
uniform vec4 mask;

#!VARYINGS
varying vec2 vUv;
varying vec2 vWorldPos;

#!SHADER: GLUIObjectMask.vs
void main() {
    vUv = uv;
    vWorldPos = (modelMatrix * vec4(position.xy, 0.0, 1.0)).xy;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: GLUIObjectMask.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.a *= uAlpha;

    if (vWorldPos.x > mask.x + mask.z) discard;
    if (vWorldPos.x < mask.x) discard;
    if (vWorldPos.y > mask.y) discard;
    if (vWorldPos.y < mask.y - mask.w) discard;
}{@}Line3D.glsl{@}#!ATTRIBUTES

attribute vec3 previous;
attribute vec3 next;
attribute float side;
attribute float width;
attribute float lineIndex;
attribute vec2 uv2;

#!UNIFORMS

uniform float uWidth;
uniform float uOpacity;
uniform vec3 uColor;

#!VARYINGS
varying float vLineIndex;
varying vec2 vUv;
varying vec2 vUv2;
varying vec3 vColor;
varying float vOpacity;
varying float vWidth;
varying float vDist;


#!SHADER: Vertex

vec2 when_eq(vec2 x, vec2 y) {
  return 1.0 - abs(sign(x - y));
}

//params

vec2 fix(vec4 i, float aspect) {
    vec2 res = i.xy / i.w;
    res.x *= aspect;
    return res;
}

void main() {
    float aspect = resolution.x / resolution.y;

    vUv = uv;
    vUv2 = uv2;
    vLineIndex = lineIndex;
    vColor = uColor;
    vOpacity = uOpacity;

    vec3 pos = position;
    vec3 prevPos = previous;
    vec3 nextPos = next;
    float lineWidth = 1.0;

    mat4 m = projectionMatrix * modelViewMatrix;
    vec4 finalPosition = m * vec4(pos, 1.0);
    vec4 pPos = m * vec4(prevPos, 1.0);
    vec4 nPos = m * vec4(nextPos, 1.0);

    vec2 currentP = fix(finalPosition, aspect);
    vec2 prevP = fix(pPos, aspect);
    vec2 nextP = fix(nPos, aspect);

    // float w = uWidth * uLineWidth * width * pressure * lineWidth;
    // float w = uWidth * uLineWidth * width * lineWidth;
    float w = 0.2;
    w *= sin(vUv2.x * 3.0 + time * 2.0 + vUv.x) * 0.3 + 0.7;
    vWidth = w;

    vec2 dirNC = normalize(currentP - prevP);
    vec2 dirPC = normalize(nextP - currentP);

    vec2 dir1 = normalize(currentP - prevP);
    vec2 dir2 = normalize(nextP - currentP);
    vec2 dirF = normalize(dir1 + dir2);

    vec2 dirM = mix(dirPC, dirNC, when_eq(nextP, currentP));
    vec2 dir = mix(dirF, dirM, clamp(when_eq(nextP, currentP) + when_eq(prevP, currentP), 0.0, 1.0));

    vec2 normal = vec2(-dir.y, dir.x);
    normal.x /= aspect;
    normal *= 0.5 * w;

    vDist = finalPosition.z / 10.0;

    finalPosition.xy += normal * side;

    gl_Position = finalPosition;
}

#!SHADER: Fragment

#require(rgb2hsv.fs)

void main() {

    // TODO: d relative to vWidth
//    float d = 0.1;
//    float d = 1.0 - vWidth vDist;
    float d = (1.0 / (5.0 * vWidth + 1.0)) * 0.1 * (vDist * 5.0 + 0.5);
//    float d = (1.0 / (vWidth + 1)) * 1.0 * (vDist * 0.9 + 0.1);
    float smoothEdge = smoothstep(0.0, 0.0 + d, vUv.y) * smoothstep(1.0, 1.0 - d, vUv.y);

    vec2 uvButt = vec2(0.0, vUv.y);
    float buttLength = 0.5 * vWidth;
    uvButt.x = min(0.5, vUv2.x / buttLength) + (0.5 - min(0.5, (vUv2.y - vUv2.x) / buttLength));
    float round = length(uvButt - 0.5);

    vec3 rgb = vec3(0.0, 0.0, 1.0);
    vec3 hsv = rgb2hsv(rgb);
    hsv.x += vUv2.x * 0.3 + time + vLineIndex * 2.1;
    hsv.y = 0.8;
    hsv.z += sin(vUv2.x + time * 2.0+ vLineIndex * 2.1);
    rgb = hsv2rgb(hsv);
    rgb += sin(vUv2.x + time * 2.0+ vLineIndex * 2.1) * 0.2 + 0.2;


    // gl_FragColor.rgb = vColor;
    gl_FragColor.rgb = rgb;
    gl_FragColor.a = smoothstep(0.5, 0.5 - d, round);
    // gl_FragColor.a = 1.0;
    gl_FragColor.a *= vOpacity;
}
{@}luma.fs{@}float luma(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

float luma(vec4 color) {
  return dot(color.rgb, vec3(0.299, 0.587, 0.114));
}{@}matcap.vs{@}vec2 reflectMatcap(vec3 position, mat4 modelViewMatrix, mat3 normalMatrix, vec3 normal) {
    vec4 p = vec4(position, 1.0);
    
    vec3 e = normalize(vec3(modelViewMatrix * p));
    vec3 n = normalize(normalMatrix * normal);
    vec3 r = reflect(e, n);
    float m = 2.0 * sqrt(
        pow(r.x, 2.0) +
        pow(r.y, 2.0) +
        pow(r.z + 1.0, 2.0)
    );
    
    vec2 uv = r.xy / m + .5;
    
    return uv;
}

vec2 reflectMatcap(vec3 position, mat4 modelViewMatrix, vec3 normal) {
    vec4 p = vec4(position, 1.0);
    
    vec3 e = normalize(vec3(modelViewMatrix * p));
    vec3 n = normalize(normal);
    vec3 r = reflect(e, n);
    float m = 2.0 * sqrt(
                         pow(r.x, 2.0) +
                         pow(r.y, 2.0) +
                         pow(r.z + 1.0, 2.0)
                         );
    
    vec2 uv = r.xy / m + .5;
    
    return uv;
}

vec2 reflectMatcap(vec4 mvPos, vec3 normal) {
    vec3 e = normalize(vec3(mvPos));
    vec3 n = normalize(normal);
    vec3 r = reflect(e, n);
    float m = 2.0 * sqrt(
                         pow(r.x, 2.0) +
                         pow(r.y, 2.0) +
                         pow(r.z + 1.0, 2.0)
                         );

    vec2 uv = r.xy / m + .5;

    return uv;
}{@}MouseFlowMapBlend.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D uTexture;
uniform sampler2D uStamp;
uniform float uSpeed;
uniform float uFirstDraw;

#!VARYINGS

varying vec2 vUv;

#!SHADER: MouseFlowMapBlend.vs

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: MouseFlowMapBlend.fs

vec3 blend(vec3 base, vec3 blend, float opacity) {
    return blend + (base * (1.0 - opacity));
}

#require(range.glsl)

void main() {
    vec3 prev = texture2D(uTexture, vUv).rgb;
    prev = prev * 2.0 - 1.0;
    float amount = crange(length(prev.rg), 0.0, 0.4, 0.0, 1.0);
    amount = 0.5 + 0.48 * (1.0 - pow(1.0 - amount, 3.0));
    prev *= amount;
    prev = prev * 0.5 + 0.5;

    // blue not used
    prev.b = 0.5;

    vec4 tex = texture2D(uStamp, vUv);
    gl_FragColor.rgb = blend(prev, tex.rgb, tex.a);

    // Force a grey on first draw to have init values
    gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(0.5, 0.5, 0.5), uFirstDraw);
    gl_FragColor.a = 1.0;
}
{@}MouseFlowMapStamp.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform vec2 uVelocity;
uniform float uFalloff;
uniform float uAlpha;

#!VARYINGS

varying vec2 vUv;

#!SHADER: MouseFlowMapStamp.vs

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: MouseFlowMapStamp.fs

void main() {
    gl_FragColor.rgb = vec3(uVelocity * 0.5 + 0.5, 1.0);
    gl_FragColor.a = smoothstep(0.5, 0.499 - (uFalloff * 0.499), length(vUv - 0.5)) * uAlpha;
}
{@}flowmap.fs{@}float getFlowMask(sampler2D map, vec2 uv) {
    vec2 flow = texture2D(map, uv).rg;
    return length(flow.rg * 2.0 - 1.0);
}

vec2 getFlow(sampler2D map, vec2 uv) {
    return texture2D(map, uv).rg * 2.0 - 1.0;
}{@}NeutrinoBase.fs{@}#require(neutrino.glsl)

uniform float lerp;

//params

void main() {
    vec2 uv = getUV();
    vec4 pos = getData4(tInput, uv);
    vec3 index = getData(tIndices, uv);
    vec4 activePos = getData4(tActive, uv);
    vec4 attribs = getData4(tAttribs, uv);

    float CHAIN = index.x;
    float LINE = index.y;

    if (pos.w > 0.9) { //head of the chain

        if (activePos.a < 0.01) {
            gl_FragColor = pos; //still
            return;
        }

        if (activePos.a > 0.7 && activePos.a < 0.8 || activePos.a > 0.05 && activePos.a < 0.15) { //if its in the initial state
            pos.xyz = activePos.xyz;
            gl_FragColor = pos;
            return;
        }

        if (activePos.a > 0.25) { //OK to move!
        //main
        }

    } else {

        float followIndex = getIndex(LINE, CHAIN-1.0);
        vec3 followPos = getData(tInput, getUVFromIndex(followIndex));

        float headIndex = getIndex(LINE, 0.0);
        vec4 headActive = getData4(tActive, getUVFromIndex(headIndex));

        if (headActive.a < 0.01) { //still
            gl_FragColor = pos;
            return;
        }

        if (headActive.a > 0.7 && headActive.a < 0.8 || headActive.a > 0.05 && headActive.a < 0.15) { //still in the init state
            pos.xyz = headActive.xyz;
            gl_FragColor = pos;
            return;
        }

        pos.xyz += (followPos - pos.xyz) * lerp;

    }

    gl_FragColor = pos;
}{@}NeutrinoTube.glsl{@}#!ATTRIBUTES
attribute float angle;
attribute vec2 tuv;
attribute float cIndex;
attribute float cNumber;

#!UNIFORMS
uniform sampler2D tPositions;
uniform sampler2D tLife;
uniform float radialSegments;
uniform float thickness;
uniform float taper;

#!VARYINGS
varying float vLength;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vPos;
varying vec2 vUv;
varying float vIndex;
varying float vLife;

#!SHADER: NeutrinoTube.vs

#define PI 3.1415926535897932384626433832795

//neutrinoparams

#require(neutrino.glsl)
#require(range.glsl)
#require(conditionals.glsl)

void createTube(vec2 volume, out vec3 offset, out vec3 normal) {
    float posIndex = getIndex(cNumber, cIndex);
    float nextIndex = getIndex(cNumber, cIndex + 1.0);

    vLength = cIndex/(lineSegments-1.0);
    vIndex = cIndex;

    vec3 current = texture2D(tPositions, getUVFromIndex(posIndex)).xyz;
    vec3 next = texture2D(tPositions, getUVFromIndex(nextIndex)).xyz;

    vec3 T = normalize(next - current);
    vec3 B = normalize(cross(T, next + current));
    vec3 N = -normalize(cross(B, T));

    float tubeAngle = angle;
    float circX = cos(tubeAngle);
    float circY = sin(tubeAngle);

    volume *= mix(crange(vLength, 1.0 - taper, 1.0, 1.0, 0.0) * crange(vLength, 0.0, taper, 0.0, 1.0), 1.0, when_eq(taper, 0.0));

    normal.xyz = normalize(B * circX + N * circY);
    offset.xyz = current + B * volume.x * circX + N * volume.y * circY;
}

void main() {
    float headIndex = getIndex(cNumber, 0.0);
    float life = texture2D(tLife, getUVFromIndex(headIndex)).z;
    vLife = life;

    float scale = 1.0;
    //neutrinovs
    vec2 volume = vec2(thickness * 0.065 * scale);

    vec3 transformed;
    vec3 objectNormal;
    createTube(volume, transformed, objectNormal);

    vec3 transformedNormal = normalMatrix * objectNormal;
    vNormal = normalize(transformedNormal);
    vUv = tuv.yx;

    vec3 pos = transformed;
    vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0);
    vViewPosition = -mvPosition.xyz;
    vPos = pos;
    gl_Position = projectionMatrix * mvPosition;
}

#!SHADER: NeutrinoTube.fs
void main() {
    gl_FragColor = vec4(1.0);
}{@}neutrino.glsl{@}uniform sampler2D tIndices;
uniform sampler2D tActive;
uniform sampler2D tAttribs;
uniform float textureSize;
uniform float lineSegments;

vec2 getUVFromIndex(float index) {
    float size = textureSize;
    vec2 uv = vec2(0.0);
    float p0 = index / size;
    float y = floor(p0);
    float x = p0 - y;
    uv.x = x;
    uv.y = y / size;
    return uv;
}

float getIndex(float line, float chain) {
    return (line * lineSegments) + chain;
}{@}range.glsl{@}float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    float oldRange = oldMax - oldMin;
    float newRange = newMax - newMin;
    return (((oldValue - oldMin) * newRange) / oldRange) + newMin;
}

float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax), min(newMax, newMin), max(newMin, newMax));
}{@}rgb2hsv.fs{@}vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}{@}rgbshift.fs{@}vec4 getRGB(sampler2D tDiffuse, vec2 uv, float angle, float amount) {
    vec2 offset = vec2(cos(angle), sin(angle)) * amount;
    vec4 r = texture2D(tDiffuse, uv + offset);
    vec4 g = texture2D(tDiffuse, uv);
    vec4 b = texture2D(tDiffuse, uv - offset);
    return vec4(r.r, g.g, b.b, g.a);
}{@}SceneLayout.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tMask;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;

#!SHADER: SceneLayout.vs
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: SceneLayout.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.a *= texture2D(tMask, vUv).r * uAlpha;
}{@}ScreenQuad.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;

#!SHADER: ScreenQuad.vs
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: ScreenQuad.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
}{@}simplenoise.glsl{@}const float PI = 3.141592653589793;
const float TAU = 6.283185307179586;

float getNoise(vec2 uv, float time) {
    float x = uv.x * uv.y * time * 1000.0;
    x = mod(x, 13.0) * mod(x, 123.0);
    float dx = mod(x, 0.01);
    float amount = clamp(0.1 + dx * 100.0, 0.0, 1.0);
    return amount;
}

highp float random(vec2 co) {
    highp float a = 12.9898;
    highp float b = 78.233;
    highp float c = 43758.5453;
    highp float dt = dot(co.xy, vec2(a, b));
    highp float sn = mod(dt, 3.14);
    return fract(sin(sn) * c);
}

float cnoise(vec3 v) {
    float t = v.z * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += range(sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0) + sin(v.x * -3.5 / s + t * 4.0) + sin(v.x * -2.5 / s + t * 7.1), -1.0, 1.0, -0.3, 0.3);
    noise += range(sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0) + sin(v.y * 2.6 / s + t * 8.0) + sin(v.y * -2.6 / s + t * 4.5), -1.0, 1.0, -0.3, 0.3);
    return noise;
}

float cnoise(vec2 v) {
    float t = v.x * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += range(sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0) + sin(v.x * -3.5 / s + t * 4.0) + sin(v.x * -2.5 / s + t * 7.1), -1.0, 1.0, -0.3, 0.3);
    noise += range(sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0) + sin(v.y * 2.6 / s + t * 8.0) + sin(v.y * -2.6 / s + t * 4.5), -1.0, 1.0, -0.3, 0.3);
    return noise;
}{@}simplex3d.glsl{@}// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
    return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v) {
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    i = mod289(i);
    vec4 p = permute( permute( permute(
          i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
        + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
        + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
}

//float surface(vec3 coord) {
//    float n = 0.0;
//    n += 1.0 * abs(snoise(coord));
//    n += 0.5 * abs(snoise(coord * 2.0));
//    n += 0.25 * abs(snoise(coord * 4.0));
//    n += 0.125 * abs(snoise(coord * 8.0));
//    float rn = 1.0 - n;
//    return rn * rn;
//}{@}transformUV.glsl{@}vec2 transformUV(vec2 uv, float a[9]) {

    // Convert UV to vec3 to apply matrices
	vec3 u = vec3(uv, 1.0);

    // Array consists of the following
    // 0 translate.x
    // 1 translate.y
    // 2 skew.x
    // 3 skew.y
    // 4 rotate
    // 5 scale.x
    // 6 scale.y
    // 7 origin.x
    // 8 origin.y

    // Origin before matrix
    mat3 mo1 = mat3(
        1, 0, -a[7],
        0, 1, -a[8],
        0, 0, 1);

    // Origin after matrix
    mat3 mo2 = mat3(
        1, 0, a[7],
        0, 1, a[8],
        0, 0, 1);

    // Translation matrix
    mat3 mt = mat3(
        1, 0, -a[0],
        0, 1, -a[1],
    	0, 0, 1);

    // Skew matrix
    mat3 mh = mat3(
        1, a[2], 0,
        a[3], 1, 0,
    	0, 0, 1);

    // Rotation matrix
    mat3 mr = mat3(
        cos(a[4]), sin(a[4]), 0,
        -sin(a[4]), cos(a[4]), 0,
    	0, 0, 1);

    // Scale matrix
    mat3 ms = mat3(
        1.0 / a[5], 0, 0,
        0, 1.0 / a[6], 0,
    	0, 0, 1);

	// apply translation
   	u = u * mt;

	// apply skew
   	u = u * mh;

    // apply rotation relative to origin
    u = u * mo1;
    u = u * mr;
    u = u * mo2;

    // apply scale relative to origin
    u = u * mo1;
    u = u * ms;
    u = u * mo2;

    // Return vec2 of new UVs
    return u.xy;
}

vec2 rotateUV(vec2 uv, float r, vec2 origin) {
    vec3 u = vec3(uv, 1.0);

    mat3 mo1 = mat3(
        1, 0, -origin.x,
        0, 1, -origin.y,
        0, 0, 1);

    mat3 mo2 = mat3(
        1, 0, origin.x,
        0, 1, origin.y,
        0, 0, 1);

    mat3 mr = mat3(
        cos(r), sin(r), 0,
        -sin(r), cos(r), 0,
        0, 0, 1);

    u = u * mo1;
    u = u * mr;
    u = u * mo2;

    return u.xy;
}

vec2 rotateUV(vec2 uv, float r) {
    return rotateUV(uv, r, vec2(0.5));
}

vec2 translateUV(vec2 uv, vec2 translate) {
    vec3 u = vec3(uv, 1.0);
    mat3 mt = mat3(
        1, 0, -translate.x,
        0, 1, -translate.y,
        0, 0, 1);

    u = u * mt;
    return u.xy;
}

vec2 scaleUV(vec2 uv, vec2 scale, vec2 origin) {
    vec3 u = vec3(uv, 1.0);

    mat3 mo1 = mat3(
        1, 0, -origin.x,
        0, 1, -origin.y,
        0, 0, 1);

    mat3 mo2 = mat3(
        1, 0, origin.x,
        0, 1, origin.y,
        0, 0, 1);

    mat3 ms = mat3(
        1.0 / scale.x, 0, 0,
        0, 1.0 / scale.y, 0,
        0, 0, 1);

    u = u * mo1;
    u = u * ms;
    u = u * mo2;
    return u.xy;
}

vec2 scaleUV(vec2 uv, vec2 scale) {
    return scaleUV(uv, scale, vec2(0.5));
}
{@}tunnelblur.fs{@}#require(rgbshift.fs)

vec4 tunnelBlur(sampler2D tDiffuse, vec2 uv, float sampleDist, float strength) {
    float samples[10];
    samples[0] = -0.08;
    samples[1] = -0.05;
    samples[2] = -0.03;
    samples[3] = -0.02;
    samples[4] = -0.01;
    samples[5] =  0.01;
    samples[6] =  0.02;
    samples[7] =  0.03;
    samples[8] =  0.05;
    samples[9] =  0.08;

    vec2 dir = 0.5 - uv;
    float dist = sqrt(dir.x*dir.x + dir.y*dir.y);
    dir = dir / dist;

    vec4 texel = texture2D(tDiffuse, uv);
    vec4 sum = texel;

    for (int i = 0; i < 10; i++) {
        sum += texture2D(tDiffuse, uv + dir * samples[i] * sampleDist);
    }

    sum *= 1.0/10.0;
    float t = clamp(dist * strength, 0.0, 1.0);

    return mix(texel, sum, t);
}{@}UnrealBloom.fs{@}uniform sampler2D tUnrealBloom;

void applyUnrealBloom(inout vec4 texel, vec2 uv) {
    texel.rgb += texture2D(tUnrealBloom, uv).rgb;
}{@}UnrealBloomComposite.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D blurTexture1;
uniform float bloomStrength;
uniform float bloomRadius;
uniform vec3 bloomTintColor;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex.vs
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment.fs

float lerpBloomFactor(const in float factor) {
    float mirrorFactor = 1.2 - factor;
    return mix(factor, mirrorFactor, bloomRadius);
}

void main() {
    gl_FragColor = bloomStrength * (lerpBloomFactor(1.0) * vec4(bloomTintColor, 1.0) * texture2D(blurTexture1, vUv));
}{@}UnrealBloomGaussian.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D colorTexture;
uniform vec2 texSize;
uniform vec2 direction;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex.vs
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment.fs

float gaussianPdf(in float x, in float sigma) {
    return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

void main() {
    vec2 invSize = 1.0 / texSize;
    float fSigma = float(SIGMA);
    float weightSum = gaussianPdf(0.0, fSigma);
    vec3 diffuseSum = texture2D( colorTexture, vUv).rgb * weightSum;
    for(int i = 1; i < KERNEL_RADIUS; i ++) {
        float x = float(i);
        float w = gaussianPdf(x, fSigma);
        vec2 uvOffset = direction * invSize * x;
        vec3 sample1 = texture2D( colorTexture, vUv + uvOffset).rgb;
        vec3 sample2 = texture2D( colorTexture, vUv - uvOffset).rgb;
        diffuseSum += (sample1 + sample2) * w;
        weightSum += 2.0 * w;
    }
    gl_FragColor = vec4(diffuseSum/weightSum, 1.0);
}{@}UnrealBloomLuminosity.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tDiffuse;
uniform vec3 defaultColor;
uniform float defaultOpacity;
uniform float luminosityThreshold;
uniform float smoothWidth;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex.vs
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment.fs

#require(luma.fs)

void main() {
    vec4 texel = texture2D(tDiffuse, vUv);
    float v = luma(texel.xyz);
    vec4 outputColor = vec4(defaultColor.rgb, defaultOpacity);
    float alpha = smoothstep(luminosityThreshold, luminosityThreshold + smoothWidth, v);
    gl_FragColor = mix(outputColor, texel, alpha);
}{@}UnrealBloomPass.fs{@}#require(UnrealBloom.fs)

void main() {
    gl_FragColor = texture2D(tDiffuse, vUv);
    applyUnrealBloom(gl_FragColor, vUv);
}{@}ProductShader.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMatCap;
uniform sampler2D tAO;
uniform sampler2D tEnv;
uniform sampler2D tVariables;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform float uToneScale;
uniform float uHueShiftScale;
uniform float uReflectionScale;
uniform float uAOScale;
uniform float uToneShift;
uniform float uValueScale;
uniform float uInactive;
uniform float uVariablesFade;
uniform float uVariablesIndex;
uniform float uFamilyIndex;

#!VARYINGS

varying vec2 vUv;
varying vec2 vUvMatcap;
varying vec3 vPos;
varying vec4 vMPos;
varying vec3 vNormal;

#!SHADER: Vertex

#require(matcap.vs)

void main() {
    vUv = uv;
    vPos = position;
    vNormal = normalize(normalMatrix * normal);
    vMPos = modelMatrix * vec4(position, 1.0);
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vUvMatcap = reflectMatcap(mvPos, vNormal);
    gl_Position = projectionMatrix * mvPos;
}

#!SHADER: Fragment

#require(rgb2hsv.fs)
#require(envmap.glsl)

void main() {
    vec3 normal = normalize(vNormal);
    float tone = mix(1.0, texture2D(tMatCap, vUvMatcap).g, 0.5);
    float ao = texture2D(tAO, vUv).g;
    tone *= ao;

    tone -= 0.5;
    tone *= uToneScale;
    tone += 0.5;
    
    tone += uToneShift;

    vec3 rgb = mix(uColor2, uColor1, tone);

    // vec3 rgb = uColor;
    vec3 hsv = rgb2hsv(rgb);
    hsv.x += vPos.x * 0.02 * uHueShiftScale + uFamilyIndex * 0.02;
    hsv.z += tone * uValueScale;
    rgb = hsv2rgb(hsv);

    vec4 env = envmap(vMPos, normal, tEnv);
    float refl = env.g * env.g;
    rgb += refl * 0.2 * uReflectionScale;
    
    rgb *= mix(1.0, ao, uAOScale);


    // uVariablesIndex
    // 0, 1, 2 = coverage - low, average, high
    // 3, 4, 5 = packaging-space - low, average, high
    // 6 = top-face
    // 7, 8, 9 = pass-through - poor, average, best
    float topFace = smoothstep(0.2, 0.1, abs(uVariablesIndex - 6.0));
    float coverage = smoothstep(2.3, 2.2, uVariablesIndex);
    float coverageRange = (1.0 - 0.7 * min(1.0, uVariablesIndex / 2.0)) * 0.8;
    float packaging = smoothstep(2.7, 2.8, uVariablesIndex) * smoothstep(5.3, 5.2, uVariablesIndex);
    float packagingRange = max(0.0, min(1.0, (uVariablesIndex - 3.0) / 2.0)) * 0.1;
    float passThrough = smoothstep(6.7, 6.8, uVariablesIndex);
    float passThroughRange = max(0.0, min(1.0, (uVariablesIndex - 7.0) / 2.0)) * 0.2;

    vec2 uvVariables = vUv;
    uvVariables *= mix(1.0, 5.0, topFace);


    vec3 variablesTex = texture2D(tVariables, uvVariables).rgb;

    // Coverage
    vec3 rgbCoverage = mix(rgb, mix(rgb, vec3(1.0), 0.7), smoothstep(coverageRange - 0.01, coverageRange, variablesTex.g));
    variablesTex = mix(variablesTex, rgbCoverage, coverage);

    // Packaging Space
    vec3 rgbPackaging = hsv2rgb(vec3( min(0.7, max(0.0, (1.0 - variablesTex.g * 1.2) * 0.75 + packagingRange)), 1.0, 1.0));
    variablesTex = mix(variablesTex, rgbPackaging, packaging);

    // Top face
    variablesTex = mix(variablesTex, rgb + variablesTex.g, topFace);

    // Pass through
    vec3 rgbPass = mix(rgb, hsv2rgb(vec3(1.0 + passThroughRange * 2.5, 1.0, 1.0)), smoothstep(0.8 - passThroughRange, 0.7 - passThroughRange, variablesTex.g));
    variablesTex = mix(variablesTex, rgbPass, passThrough);


    rgb = mix(rgb, variablesTex, uVariablesFade);

    float alpha = 1.0;

    alpha = mix(alpha, 0.05, smoothstep(0.0, 1.0, uInactive));
    // rgb = mix(rgb, rgb * 0.1, uInactive);
    rgb = mix(rgb, vec3(0.1), smoothstep(0.0, 1.0, uInactive) );

    gl_FragColor = vec4(rgb, alpha);
}
{@}Layer.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tDiffuse;
uniform sampler2D tAlpha;
uniform float uThickness;
uniform float uTransition;
uniform float uOffset;
uniform float uDir;

#!VARYINGS

varying vec2 vUv;
varying vec3 vPos;
varying vec3 vNormal;
varying float vFlip;

#!SHADER: Vertex

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

void main() {
    vUv = uv;
    vec3 pos = position;
    vNormal = normalize(normalMatrix * normal);

    // Scale to required thickness
    float bottom = step(pos.y, -0.5);
    pos.y *= mix(1.0, uThickness, bottom);

    vPos = pos;

    // Add sine waves
    pos.y += sin((pos.x + pos.z) * 6.28 * 0.5 + time + uOffset) * 0.05;
    pos.x += sin((pos.z) * 6.28 * 0.5 + time + uOffset) * 0.02;

    // Transition fold
    // float t = mix(1.0, -1.0, sin(time) * 0.5 + 0.5);
    float t = mix(1.0, -1.0, uTransition);
    float flip = mix(1.0, -1.0, uDir);
    vFlip = flip;
    pos.x -= 0.5 - uDir;
    float angle = smoothstep(t + 1.0, t - 1.0, position.x * flip + 0.5) * 0.8 * flip;
    rotate2D(pos.xy, angle);
    pos.x += 0.5 - uDir;

    pos.y += (1.0 - uTransition) * 0.2;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

void main() {
    vec3 rgb = texture2D(tDiffuse, vUv).rgb;

    // Bit of color gradient
    rgb += length(vPos - vec3(-0.5, 0.0, -0.5)) * vec3(0.3, 0.2, 0.4) * 0.5;

    // bit of shadowing
    rgb *= mix(0.5, 1.0, smoothstep(-1.0, 0.5, dot(normalize(vec3(1.0, 1.0, 0.0)), vNormal)));

    float alpha = texture2D(tAlpha, vUv).g;

    // Dots to hide
    // float t = mix(1.1, 0.4, (sin(time) * 0.5 + 0.5));
    float t = mix(1.1, 0.0, uTransition);
    float spheres = length( mod(vPos * 40.0, vec3(1.0)) - 0.5 );
    alpha *= smoothstep(t - 0.1, t, spheres + (vPos.x * vFlip * 0.3) );

    if (alpha < 0.1) discard;

    gl_FragColor.rgb = rgb;
    gl_FragColor.a = 1.0;
}
{@}LayerShadow.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tAlpha;
uniform float uAlpha;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    float alpha = texture2D(tAlpha, vUv).g;
    gl_FragColor.rgb = vec3(0.0);
    gl_FragColor.a = alpha * uAlpha * smoothstep(2.0, 6.0, cameraPosition.y);
}
{@}EngineParticleFlow.fs{@}uniform float uCurlScale;
uniform float uTimeScale;
uniform float uCurlStrength;
uniform float uSpeed;


#require(curl.glsl)

void main() {
    vec3 curl = curlNoise(pos.xyz * uCurlScale + (time * uTimeScale * 0.1));
//     pos.xyz += curl * uCurlStrength * 0.1;
    //  pos.y += 0.01;
    // pos.z -= 0.03 * uSpeed;
}{@}ParticleFlow.fs{@}uniform float uCurlScale;
uniform float uTimeScale;
uniform float uCurlStrength;
uniform float uSpeed;


#require(curl.glsl)

void main() {
    vec3 curl = curlNoise(pos.xyz * uCurlScale + (time * uTimeScale * 0.1));
    // pos.xyz += curl * uCurlStrength * 0.1;
    // pos.y += 0.01;
    // pos.z -= 0.03 * uSpeed;
}{@}ParticleShader.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tLife;
uniform sampler2D tAttribs;
uniform sampler2D tColor;
uniform float uSize;
uniform float uSizeIncrease;

#!VARYINGS

varying float vLife;
varying vec4 vAttribs;
varying vec3 vColor;

#!SHADER: Vertex

void main() {
    vec3 pos = getPos();

    vLife = texture2D(tLife, position.xy).z;
    vAttribs = texture2D(tAttribs, position.xy);
    vColor = texture2D(tColor, vec2(1.0 - vLife, 0.5)).rgb;

    vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
    gl_PointSize = ((1.0 - vLife) * uSizeIncrease * vAttribs.z + 1.0) * uSize * (1.0 - 0.8 * vAttribs.x) * 0.3 * (1000.0 / length(mvPos.xyz));
    gl_Position = projectionMatrix * mvPos;
}

#!SHADER: Fragment

void main() {
    vec2 uv = gl_PointCoord;
    gl_FragColor.rgb = vColor;

    float falloff = length(uv - 0.5);
    float circle = smoothstep(0.5, 0.4, falloff) * (1.0 - 0.5 * smoothstep(0.4, 0.3, falloff));
    gl_FragColor.a = circle;

    // fade off at end
    gl_FragColor.a *= smoothstep(0.0, 0.1, vLife);
    

    // Get randomly fainter as aging
    gl_FragColor.a *= mix(vLife, 1.0, vAttribs.y);

}
{@}ParticlesEngine.glsl{@}#!ATTRIBUTES

attribute float spread;
attribute vec3 random;

#!UNIFORMS

uniform sampler2D tColor;

#!VARYINGS

varying float vSpread;
varying float vLife;

#!SHADER: Vertex

void main() {
    vSpread = spread;

    vec3 pos = position;

    float cycle = 1.0;
    float t = max(0.0, mod(vSpread * 1.0 - time * 1.0, cycle) - (cycle - 1.0));
    vLife = smoothstep(0.8, 1.0, t);

    pos += normal * (1.0 - pow(vLife, 2.0)) * (random.z * 0.4 + 0.6) * (2.0 - vSpread) * 0.05;

    pos.x += sin(time * 2.0 * random.x + 3.14 * random.y) * random.z * 0.1;
    pos.y += sin(time * 2.0 * random.y + 3.14 * random.z) * random.x * 0.1;
    pos.z += sin(time * 2.0 * random.z + 3.14 * random.x) * random.y * 0.1;

    vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    gl_PointSize = (2.0 - vLife) * 2.0 * (random.x * 0.5 + 0.7) * 0.05 * (1000.0 / length(mvPos.xyz)) * 0.5;
    // gl_PointSize = (2.0 - vLife) * 2.0 * 0.05 * (1000.0 / length(mvPos.xyz)) * 0.5;
}

#!SHADER: Fragment

void main() {
    vec2 uv = gl_PointCoord;

    float gradient = vLife;

    gl_FragColor.rgb = texture2D(tColor, vec2(vSpread, 0.5)).rgb;
    gl_FragColor.rgb += gradient * 0.2;

    float fade = smoothstep(1.0, 0.5, vSpread);

    float circle = smoothstep(0.5, 0.49, length(uv - 0.5));

    float alpha = gradient * fade;
    alpha *= circle;
    
    if (alpha < 0.01) discard;
    gl_FragColor.a = alpha;
}
{@}Room.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS

varying vec2 vUv;
varying vec3 vPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    // gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    vec3 color = vec3(0.0);

    // Wall corners
    float dist = length(vUv - 0.5);
    float gradient = smoothstep(2.5, 0.3, dist);

    // Screen Vignette
    vec2 uv = gl_FragCoord.xy / resolution;
    dist = length(uv - 0.5);
    gradient *= smoothstep(1.3, 0.2, dist);

    // Lines

    float width = 0.01;
    vec3 sdf = abs(mod(vPos * 2.0, vec3(1.0)) - 0.5);
    float line = 0.0;
    line = max(line, smoothstep(width + 0.001, width, sdf.x));
    line = max(line, smoothstep(width + 0.001, width, sdf.y));
    line = max(line, smoothstep(width + 0.001, width, sdf.z));

    color += gradient;
    color -= line * 0.05 * smoothstep(0.4, 0.0, dist);


    gl_FragColor.rgb = color;
    gl_FragColor.a = 1.0;
}
{@}RoomShadow.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMap;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    float shadow = (1.0 - texture2D(tMap, vUv).g) * 0.5;
    gl_FragColor.rgb = vec3(0.0);
    gl_FragColor.a = shadow * smoothstep(2.0, 6.0, cameraPosition.y);
}
{@}RoomTransition.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform float uTransition;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    vec3 color = vec3(0.0);

    // float t = sin(time) * 0.5 + 0.5;
    float t = uTransition;

    vec2 center = vUv - 0.5;
    float dist = length(center);
    float angle = atan(center.x, center.y);
    dist *= smoothstep(1.0, 0.0, abs(angle)) * 1.0 * (0.7 - abs(t - 0.6)) + 1.0;
    dist /= smoothstep(0.0, 1.0, abs(center.x)) * 0.3 + 1.0;

    float falloff = 0.001;
    float mask = smoothstep(t, t - falloff, dist / 1.0);

    color += mask;

    gl_FragColor.rgb = color;
    gl_FragColor.a = 1.0;
}
{@}BackgroundParticlesFlow.fs{@}uniform sampler2D tOrigin;
uniform sampler2D tAttribs;
uniform mat4 uProjMatrix;
uniform mat3 uProjNormalMatrix;
uniform vec2 uProjResolution;
uniform sampler2D uFlowMap;

uniform vec3 uCenterPosition;
uniform float uFlowSpeed;
uniform float uMaxRange;
uniform float uMaxHeight;

#require(range.glsl)

vec2 frag_coord(vec4 glPos) {
    return ((glPos.xyz / glPos.w) * 0.5 + 0.5).xy;
}

float when_gt(float x, float y) {
  return max(sign(x - y), 0.0);
}

float when_lt(float x, float y) {
  return max(sign(y - x), 0.0);
}


void main() {
    vec2 uv = getUV();
    vec3 pos = getData(tInput, uv);
    vec3 origin = getData(tOrigin, uv);
    vec4 attribs = getData4(tAttribs, uv);

    // Convert pos into screenspace
    vec4 mvpPos = uProjMatrix * vec4(pos, 1.0);
    vec2 screenPos = frag_coord(mvpPos);

    // Get flowmap value
    vec2 flow = texture2D(uFlowMap, screenPos).xy * 2.0 - 1.0;
    flow.x *= smoothstep(0.0, 0.3, abs(flow.x));
    flow.y *= smoothstep(0.0, 0.3, abs(flow.y));
    flow.y *= -1.0;

    // Convert flowmap back into worldspace
    vec3 flowVel = uProjNormalMatrix * vec3(flow, 0.0) * uFlowSpeed;


    // Keep particles near rover
    vec3 dist = pos + uCenterPosition;
    float max = uMaxRange;
    pos.x += when_gt(dist.x, max) * max * -2.0;
    pos.x += when_lt(dist.x, -max) * max * 2.0;

    pos.z += when_gt(dist.z, max) * max * -2.0;
    pos.z += when_lt(dist.z, -max) * max * 2.0;

    float yMax = uMaxHeight;
    pos.y += when_gt(dist.y, yMax * 0.8) * -yMax;
    pos.y += when_lt(dist.y, -yMax * 0.2) * yMax;


    pos += flowVel;


    vec3 sway = vec3(0.0);
    sway.x = sin(time * attribs.x + attribs.y * 6.3) * 0.01;
    sway.z = sin(time * attribs.z + attribs.x * 6.3) * 0.01;


    pos.y -= 0.01 * attribs.x;
    pos += sway;

    gl_FragColor = vec4(pos, 1.0);
}{@}SceneParticles.glsl{@}#!ATTRIBUTES
attribute vec4 attribs;

#!UNIFORMS
uniform float uSize;
uniform sampler2D tPos;
uniform sampler2D tMap;
uniform vec3 uPosition;

#!VARYINGS

varying vec4 vAttribs;
varying vec4 vMPos;

#!SHADER: Vertex

float when_gt(float x, float y) {
  return max(sign(x - y), 0.0);
}

float when_lt(float x, float y) {
  return max(sign(y - x), 0.0);
}

void main() {
    vAttribs = attribs;
    vec3 pos = texture2D(tPos, position.xy).xyz;
    vMPos = modelViewMatrix * vec4(pos, 1.0);
    vec4 mVPos = modelViewMatrix * vec4(pos, 1.0);

    gl_PointSize = uSize * attribs.x * (1000.0 / length(mVPos.xyz));
    gl_Position = projectionMatrix * mVPos;
}

#!SHADER: Fragment
void main() {
    vec2 uv = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y);
    float alpha = texture2D(tMap, uv).g;
    alpha *= 0.4 * (1.0 - 0.8 * vAttribs.y);

    float dist = length(vMPos.xyz);
    alpha *= smoothstep(2.0, 5.0, dist);

    gl_FragColor = vec4(vec3(1.0), alpha * 0.4);
    // gl_FragColor = vec4(vec3(1.0), alpha * 1.0);
}{@}Engine.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform vec3 uColor;
uniform float uOpacity;

#!VARYINGS

varying vec2 vUv;
varying vec3 vPos;
varying vec3 vNormal;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    gl_FragColor.rgb = uColor + vNormal * 0.1;

    gl_FragColor.a = uOpacity;
}
{@}SoundWavesEngine.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform vec3 uColor;
uniform float uDelay;
uniform float uSpeed;
uniform float uScale;
uniform float uOpacity;
uniform float uTimeOffset;

#!VARYINGS

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vPos;
varying float vLife;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);

    vPos = position;
    vec3 pos = position;

    pos -= vec3(0.0, 0.55, 1.3);

    float delay = uDelay;
    float speed = uSpeed;
    float life = max(0.0, mod(time * speed + uTimeOffset, 1.0 + delay) - delay);
    vLife = life;

    float scale = uScale;
    pos *= life * scale;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

void main() {
    gl_FragColor.rgb = uColor + vNormal * 0.05;

    float alpha = smoothstep(1.0, 0.2, vLife);

    // alpha *= smoothstep(0.249, 0.25, abs(mod(vUv.x * 40.0, 1.0) - 0.5) );
    alpha *= 1.0 - 0.2 * smoothstep(0.249, 0.25, abs(mod(vUv.y * 10.0 + sin(vUv.x * 6.28 * 5.0 + time * 2.0) * 0.4 - time * 10.0, 1.0) - 0.5) );

    gl_FragColor.a = alpha * uOpacity;
}
{@}CarProductHotspot.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform float uAlpha;
uniform float uActive;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {

    float radius = mix(0.2, 0.5, uActive);

    vec2 center = vUv - 0.5;
    float dist = length(center);


    float spotRadius = 0.02;
    float spot = smoothstep(spotRadius, spotRadius - 0.02, dist);
    float circle = smoothstep(radius, radius - 0.02, dist);
    float edge = 0.02;
    float circleEdge = circle * smoothstep(radius - 0.02 - edge, radius - edge, dist);

    float alpha = min(1.0, spot + circleEdge + circle * 0.1) * 0.9;

    gl_FragColor.rgb = vec3(1.0);
    gl_FragColor.a = alpha * uAlpha;
}
{@}CarProductLine.glsl{@}#!ATTRIBUTES

attribute vec3 previous;
attribute vec3 next;
attribute float side;
attribute float width;
attribute float lineIndex;
attribute vec2 uv2;

#!UNIFORMS
uniform vec3 uColor;
uniform float uAlpha;
uniform float uAlpha2;

#!VARYINGS

varying float vLineIndex;
varying vec2 vUv;
varying vec2 vUv2;
varying float vDist;
varying float vRandom;

#!SHADER: Vertex


vec2 when_eq(vec2 x, vec2 y) {
    return 1.0 - abs(sign(x - y));
}

vec2 fix(vec4 i, float aspect) {
    vec2 res = i.xy / i.w;
    res.x *= aspect;
    return res;
}

void main() {
    float aspect = resolution.x / resolution.y;

    vUv = uv;
    vUv2 = uv2;
    vLineIndex = lineIndex;

    vec3 pos = position;
    vec3 prevPos = previous;
    vec3 nextPos = next;

    mat4 m = projectionMatrix * modelViewMatrix;
    vec4 finalPosition = m * vec4(pos, 1.0);
    vec4 pPos = m * vec4(prevPos, 1.0);
    vec4 nPos = m * vec4(nextPos, 1.0);
    vec2 currentP = fix(finalPosition, aspect);
    vec2 prevP = fix(pPos, aspect);
    vec2 nextP = fix(nPos, aspect);

    float w = 0.2 * width;

    vec2 dirNC = normalize(currentP - prevP);
    vec2 dirPC = normalize(nextP - currentP);
    vec2 dir1 = normalize(currentP - prevP);
    vec2 dir2 = normalize(nextP - currentP);
    vec2 dirF = normalize(dir1 + dir2);
    vec2 dirM = mix(dirPC, dirNC, when_eq(nextP, currentP));
    vec2 dir = mix(dirF, dirM, clamp(when_eq(nextP, currentP) + when_eq(prevP, currentP), 0.0, 1.0));
    vec2 normal = vec2(-dir.y, dir.x);
    normal.x /= aspect;
    normal *= 0.5 * w;
    vDist = finalPosition.z / 10.0;
    finalPosition.xy += normal * side;
    gl_Position = finalPosition;
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)

void main() {
    float t = 0.1;
    float line = smoothstep(t + 0.1, t, abs(vUv.y - 0.5));
    float halo = smoothstep(0.5, t, abs(vUv.y - 0.5));
    float alpha = max(line, halo * 0.3) * vUv.x * 1.5;

    float noise = cnoise(vec3(vUv*5.0+time*0.5, 1.0));

    vec3 color = uColor;
    color *= 1.1+noise*0.3;

    gl_FragColor.rgb = color;
    gl_FragColor.a = alpha*uAlpha*uAlpha2;
}
{@}CarProductLineButton.glsl{@}#!ATTRIBUTES

attribute vec3 previous;
attribute vec3 next;
attribute float side;
attribute float width;
attribute float lineIndex;
attribute vec2 uv2;

#!UNIFORMS
uniform vec3 uColor;
uniform float uAlpha;
uniform float uAlpha2;

#!VARYINGS

varying float vLineIndex;
varying vec2 vUv;
varying vec2 vUv2;
varying float vDist;
varying float vRandom;

#!SHADER: Vertex


vec2 when_eq(vec2 x, vec2 y) {
    return 1.0 - abs(sign(x - y));
}

vec2 fix(vec4 i, float aspect) {
    vec2 res = i.xy / i.w;
    res.x *= aspect;
    return res;
}

void main() {
    float aspect = resolution.x / resolution.y;

    vUv = uv;
    vUv2 = uv2;
    vLineIndex = lineIndex;

    vec3 pos = position;
    vec3 prevPos = previous;
    vec3 nextPos = next;

    mat4 m = projectionMatrix * modelViewMatrix;
    vec4 finalPosition = m * vec4(pos, 1.0);
    vec4 pPos = m * vec4(prevPos, 1.0);
    vec4 nPos = m * vec4(nextPos, 1.0);
    vec2 currentP = fix(finalPosition, aspect);
    vec2 prevP = fix(pPos, aspect);
    vec2 nextP = fix(nPos, aspect);

    float w = 0.2 * width;

    vec2 dirNC = normalize(currentP - prevP);
    vec2 dirPC = normalize(nextP - currentP);
    vec2 dir1 = normalize(currentP - prevP);
    vec2 dir2 = normalize(nextP - currentP);
    vec2 dirF = normalize(dir1 + dir2);
    vec2 dirM = mix(dirPC, dirNC, when_eq(nextP, currentP));
    vec2 dir = mix(dirF, dirM, clamp(when_eq(nextP, currentP) + when_eq(prevP, currentP), 0.0, 1.0));
    vec2 normal = vec2(-dir.y, dir.x);
    normal.x /= aspect;
    normal *= 0.5 * w;
    vDist = finalPosition.z / 10.0;
    finalPosition.xy += normal * side;
    gl_Position = finalPosition;
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)

void main() {

    float t = 0.1;
    float line = smoothstep(t + 0.1, t, abs(vUv.y - 0.5));
    float halo = smoothstep(0.5, t, abs(vUv.y - 0.5));
    float alpha = max(line, halo * 0.3) * vUv.x * 1.5;
    alpha *= (1.0 - smoothstep(0.5, 0.9, vUv.x));

    float noise = cnoise(vec3(vUv*5.0+time*0.5, 1.0));

    vec3 color = uColor;
    color *= 1.1+noise*0.3;

    gl_FragColor.rgb = color;
    gl_FragColor.a = alpha*uAlpha*uAlpha2;
}
{@}CarProductShader.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMatCap;
uniform sampler2D tAO;
uniform sampler2D tEnv;
uniform vec3 uColor1;
//uniform vec3 uColor2;
uniform float uSelected;
uniform float uVisible;

#!VARYINGS

varying vec2 vUv;
varying vec2 vUvMatcap;
varying vec3 vPos;
varying vec4 vMPos;
varying vec3 vNormal;

#!SHADER: Vertex

#require(matcap.vs)

void main() {
    vUv = uv;
    vPos = position;
    vNormal = normalize(normalMatrix * normal);
    vMPos = modelMatrix * vec4(position, 1.0);
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vUvMatcap = reflectMatcap(mvPos, vNormal);
    gl_Position = projectionMatrix * mvPos;
}

#!SHADER: Fragment

#require(rgb2hsv.fs)
#require(envmap.glsl)
#require(range.glsl)
#require(simplenoise.glsl)

void main() {

    // Transition alpha clip
    float fadeAlpha = mix(2.6, -3.0, uVisible);
    float fade = smoothstep(fadeAlpha - 1.0, fadeAlpha + 1.0, vPos.z - vPos.y );
    float circles = length(mod(vPos * 20.0, vec3(1.0)) - 0.5);
    if (circles > fade) discard;

    vec3 normal = normalize(vNormal);
    float matcap = texture2D(tMatCap, vUvMatcap).g;
    float ao = texture2D(tAO, vUv).g;
    float env = envmap(vMPos, normal, tEnv).g;

    float tone = mix(1.0, matcap, 0.5);
    tone *= ao * 1.5;

    float noise = cnoise(vec3(vUv*2.5+time*0.5, 1.0));

    vec3 color1 = rgb2hsv(uColor1);
    color1.x += smoothstep(0.1, 0.2, color1.x);
    color1.y *= 1.5;
    color1.z *= 0.4;
    color1 = hsv2rgb(color1);

    vec3 color2 = rgb2hsv(uColor1);
    color2.y *= 0.9;
    color2 = hsv2rgb(color2);

    vec3 color = mix(color1, color2, tone);

    color *= 1.0+noise*0.04;

    color = mix(vec3(pow(tone, 2.0) * 0.6 + 0.2), color, uSelected);

    float refl = env * env;
    color += refl * mix(0.02, 0.1, uSelected);

    gl_FragColor.rgb = color;
    gl_FragColor.a = 0.1 + 0.9*uSelected;
}
{@}Ripple.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tBackground;
uniform sampler2D tCar;
uniform sampler2D tMatcap;
#test Tests.useFluid()
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
#endtest
uniform float uSize;
uniform float uDPR;
uniform float uHole;
uniform float uStrength;
uniform float uTime;
uniform vec3 uColor;
uniform float uReport;
uniform float uFade;
uniform float uFluidMax;

#!VARYINGS

varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = vec4(position * 2.0, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(rgb2hsv.fs)

vec2 reflectMatcap(vec3 pos, vec3 normal) {
    vec3 e = normalize(vec3(pos));
    vec3 n = normalize(normal);
    vec3 r = reflect(e, n);
    float m = 2.0 * sqrt(
        pow(r.x, 2.0) +
        pow(r.y, 2.0) +
        pow(r.z + 1.0, 2.0)
        );

    vec2 uv = r.xy / m + .5;
    return uv;
}

void rotate2D(inout vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	v = m * v;
}

vec3 dither(vec3 color) {
    float grid_position = random(gl_FragCoord.xy);
    vec3 dither_shift_RGB = vec3(0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0);
    dither_shift_RGB = mix(2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position);
    return color + dither_shift_RGB;
}

void main() {
    vec2 uv = vUv;

    float strength = uStrength;

    vec3 normal = vec3(0.0, 0.0, 1.0);

    vec2 center = uv - 0.5;
    center.x *= resolution.x / resolution.y;
    rotate2D(center, -uTime*0.2);

    float dist = length(center);

    float waveScale = mix(25.0, 15.0, dist) * (1.0-uHole*0.3);
    float wave = sin(dist * waveScale - uTime);

    vec2 pixelCenter = gl_FragCoord.xy - (0.5 * resolution);
    pixelCenter /= uDPR;
    float flatMiddle = smoothstep(uSize - 100.0, uSize + 100.0, length(pixelCenter));

    vec2 dir = normalize(center);
    normal = normalize(vec3(normal.xy += dir * wave * flatMiddle, 1.0));

    vec2 uvMatcap = reflectMatcap(vec3(center, -1.0), normal);
    float matcap = texture2D(tMatcap, uvMatcap).g * 1.8 - 0.8;
    matcap *= strength;
    float lightMatcap = max(0.0, matcap);
    float darkenMatcap = min(0.0, matcap);

    vec3 background = texture2D(tBackground, uv).rgb;

    vec3 color = background;
    vec3 mixColor = mix(uColor, vec3(1.0), 0.3);

    color = mix(color, mixColor, lightMatcap);
    color = mix(color, mixColor, darkenMatcap * 0.2);

    float hole = smoothstep(uSize * uHole * 0.8, uSize * uHole * 1.12, length(pixelCenter)) * (1.0-0.2*uHole);

    #test Tests.useFluid()
    float fluidMask = texture2D(tFluidMask, gl_FragCoord.xy / resolution).r;
    vec2 fluid = texture2D(tFluid, gl_FragCoord.xy / resolution).xy * fluidMask;
    vec2 fluidCenter = vUv - 0.5;
    fluidCenter.x *= resolution.x / resolution.y;
    float ff = min(1.0, pow(length(fluid) * 0.005, 0.5)) * 0.5 * max(uFluidMax, smoothstep(0.1, 1.0, length(fluidCenter))) * mix(1.0, hole * 0.2, uFade);
    #endtest

    vec2 carUv = vUv;
    vec3 base = texture2D(tCar, carUv).rgb;
    hole = 1.0 - (1.0 - hole) * mix(1.0, 0.1, uReport);
    vec3 blend = color * hole + base * (1.0 - hole);
    blend = mix(base, blend, uFade);

    #test Tests.useFluid()
    blend = dither(mix(blend, mixColor, ff));
    #endtest

    gl_FragColor.rgb = blend;
    gl_FragColor.a = 1.0;
}
{@}Composite.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tViewport;
uniform sampler2D tCustomise;
uniform float uCustomiseTransition;
uniform float uDPR;
uniform float uFocus;
uniform float uDelta;
uniform float uAngle;
uniform float amplitude;
uniform float sp;
uniform float divider;

#!VARYINGS

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(transformUV.glsl)
#require(rgb2hsv.fs)
#require(rgbshift.fs)
#require(eases.glsl)

vec2 getRippleUV(vec2 uv) {
    float t = uCustomiseTransition * sp;
    vec2 uv2 = (vUv - 0.5) * 1.5;
    uv2.x *= resolution.x / resolution.y;
    vec2 offset = uv2 * sin(t * (length(uv2) * amplitude - sp)) / divider;
    offset.x *= smoothstep(0.0, 0.1, vUv.x);
    offset.x *= 1.0 - smoothstep(0.9, 1.0, vUv.x);
    offset.y *= smoothstep(0.0, 0.1, vUv.y);
    offset.y *= 1.0 - smoothstep(0.9, 1.0, vUv.y);
    return uv + offset * 0.5;
}

void main() {
    vec2 viewportUV = vUv;
    viewportUV = scaleUV(viewportUV, vec2(1.0 + uFocus * 0.1 + sineIn(uCustomiseTransition) * 0.45), vec2(0.5, 0.5));
    viewportUV = getRippleUV(viewportUV);

    vec2 uvCircle = (vUv - 0.5) * 1.5;
    float ratio = resolution.x / resolution.y;
    uvCircle.x *= ratio;
    uvCircle *= 1.0+cnoise(vec3(vUv*1.5, time*0.4))*0.4*(1.0-uCustomiseTransition);
    float lCircle = length(uvCircle);
    float easedTransition = sineIn(uCustomiseTransition);
    ratio = max(ratio, resolution.y / resolution.x);
    float circleTransition = smoothstep(easedTransition, easedTransition - 0.2, lCircle / (ratio * 1.2));

    vec3 viewport = texture2D(tViewport, viewportUV).rgb;

    vec3 viewportRGB = getRGB(tViewport, viewportUV, uAngle, 0.0003 + uCustomiseTransition * 0.001 + uFocus * 0.0005 * uDelta).rgb;
    viewportRGB = rgb2hsv(viewportRGB);
    viewportRGB.x = mix(viewportRGB.x, 0.1+viewportRGB.x*0.2, 0.5);
    viewportRGB = hsv2rgb(viewportRGB);
    viewport = mix(viewport, viewportRGB, uFocus*0.05*uDelta);

    vec3 customise = texture2D(tCustomise, mix(viewportUV, vUv, uCustomiseTransition)).rgb;

    vec3 color = mix(viewport, customise, circleTransition * smoothstep(0.0, 1.0, uCustomiseTransition));

    color += range(getNoise(vUv * uDPR, time), 0.0, 1.0, -1.0, 1.0) * 0.02;
    gl_FragColor.rgb = color;
    gl_FragColor.a = 1.0;
}
{@}AfterImage.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform float damp;
uniform sampler2D tOld;
uniform sampler2D tNew;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex.vs

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment.fs

vec4 when_gt(vec4 x, float y) {
    return max(sign(x - y), 0.0);
}

void main() {
    vec4 texelOld = texture2D(tOld, vUv);
	vec4 texelNew = texture2D(tNew, vUv);
	texelOld *= damp * when_gt(texelOld, 0.1);
	gl_FragColor = max(texelNew, texelOld);
}{@}AfterImagePass.fs{@}uniform sampler2D tMap;

void main() {
    gl_FragColor = texture2D(tMap, vUv);
}{@}advectionManualFilteringShader.fs{@}varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform vec2 dyeTexelSize;
uniform float dt;
uniform float dissipation;
vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
    vec2 st = uv / tsize - 0.5;
    vec2 iuv = floor(st);
    vec2 fuv = fract(st);
    vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
    vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
    vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
    vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
void main () {
    vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
    gl_FragColor = dissipation * bilerp(uSource, coord, dyeTexelSize);
    gl_FragColor.a = 1.0;
}{@}advectionShader.fs{@}varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;
void main () {
    vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
    gl_FragColor = dissipation * texture2D(uSource, coord);
    gl_FragColor.a = 1.0;
}{@}backgroundShader.fs{@}varying vec2 vUv;
uniform sampler2D uTexture;
uniform float aspectRatio;
#define SCALE 25.0
void main () {
    vec2 uv = floor(vUv * SCALE * vec2(aspectRatio, 1.0));
    float v = mod(uv.x + uv.y, 2.0);
    v = v * 0.1 + 0.8;
    gl_FragColor = vec4(vec3(v), 1.0);
}{@}clearShader.fs{@}varying vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
void main () {
    gl_FragColor = value * texture2D(uTexture, vUv);
}{@}colorShader.fs{@}uniform vec4 color;
void main () {
    gl_FragColor = color;
}{@}curlShader.fs{@}varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
    float L = texture2D(uVelocity, vL).y;
    float R = texture2D(uVelocity, vR).y;
    float T = texture2D(uVelocity, vT).x;
    float B = texture2D(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
}{@}displayShader.fs{@}varying vec2 vUv;
uniform sampler2D uTexture;
void main () {
    vec3 C = texture2D(uTexture, vUv).rgb;
    float a = max(C.r, max(C.g, C.b));
    gl_FragColor = vec4(C, a);
}{@}divergenceShader.fs{@}varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
    float L = texture2D(uVelocity, vL).x;
    float R = texture2D(uVelocity, vR).x;
    float T = texture2D(uVelocity, vT).y;
    float B = texture2D(uVelocity, vB).y;
    vec2 C = texture2D(uVelocity, vUv).xy;
//    if (vL.x < 0.0) { L = -C.x; }
//    if (vR.x > 1.0) { R = -C.x; }
//    if (vT.y > 1.0) { T = -C.y; }
//    if (vB.y < 0.0) { B = -C.y; }
    float div = 0.5 * (R - L + T - B);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
}{@}fluidBase.vs{@}varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform vec2 texelSize;

void main () {
    vUv = uv;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(position, 1.0);
}{@}gradientSubtractShader.fs{@}varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
vec2 boundary (vec2 uv) {
    return uv;
    // uv = min(max(uv, 0.0), 1.0);
    // return uv;
}
void main () {
    float L = texture2D(uPressure, boundary(vL)).x;
    float R = texture2D(uPressure, boundary(vR)).x;
    float T = texture2D(uPressure, boundary(vT)).x;
    float B = texture2D(uPressure, boundary(vB)).x;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
}{@}pressureShader.fs{@}varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
vec2 boundary (vec2 uv) {
    return uv;
    // uncomment if you use wrap or repeat texture mode
    // uv = min(max(uv, 0.0), 1.0);
    // return uv;
}
void main () {
    float L = texture2D(uPressure, boundary(vL)).x;
    float R = texture2D(uPressure, boundary(vR)).x;
    float T = texture2D(uPressure, boundary(vT)).x;
    float B = texture2D(uPressure, boundary(vB)).x;
    float C = texture2D(uPressure, vUv).x;
    float divergence = texture2D(uDivergence, vUv).x;
    float pressure = (L + R + B + T - divergence) * 0.25;
    gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
}{@}splatShader.fs{@}varying vec2 vUv;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec2 point;
uniform float radius;
uniform float canRender;

void main () {
    vec2 p = vUv - point.xy;
    p.x *= aspectRatio;
    vec3 splat = exp(-dot(p, p) / radius) * color;
    vec3 base = texture2D(uTarget, vUv).xyz;
    gl_FragColor = vec4((base*canRender) + splat, 1.0);
}{@}vorticityShader.fs{@}varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
void main () {
    float L = texture2D(uCurl, vL).x;
    float R = texture2D(uCurl, vR).x;
    float T = texture2D(uCurl, vT).x;
    float B = texture2D(uCurl, vB).x;
    float C = texture2D(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
//    force.y += 400.3;
    vec2 vel = texture2D(uVelocity, vUv).xy;
    gl_FragColor = vec4(vel + force * dt, 0.0, 1.0);
}