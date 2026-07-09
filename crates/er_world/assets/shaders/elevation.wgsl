// FastNoiseLite WGSL port + elevation compute shader
// Ported from https://github.com/Auburn/FastNoiseLite/blob/master/GLSL/FastNoiseLite.glsl
// MIT License - Copyright (c) 2023 Jordan Peck (jordan.me2@gmail.com)

const GRADIENTS_3D = array<f32, 256>(0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, 0, -1, 1, 0, -1, 1, 0, 0, 0, -1, -1, 0);

const RAND_VECS_3D = array<f32, 1024>(-0.7292737, -0.661844, 0.1735582, 0, 0.7902921, -0.5480887, -0.2739291, 0, 0.7217579, 0.6226212, -0.3023381, 0, 0.5656831, -0.8208298, -0.07900003, 0, 0.760049, -0.555598, -0.3371, 0, 0.3713946, 0.5011265, 0.7816254, 0, -0.1277062, -0.4254439, -0.8959289, 0, -0.2881561, -0.5815839, 0.7607406, 0, 0.5849561, -0.6628202, -0.4674352, 0, 0.3307171, 0.03916537, 0.9429169, 0, 0.8712122, -0.4113374, -0.2679382, 0, 0.580981, 0.7021916, 0.4115678, 0, 0.5037569, 0.6330057, -0.5878204, 0, 0.4493712, 0.6013902, 0.6606023, 0, -0.6878404, 0.09018891, -0.7202372, 0, -0.5958956, -0.646935, 0.4757977, 0, -0.5127052, 0.1946922, -0.8361987, 0, -0.9911507, -0.05410276, -0.1212153, 0, -0.2149721, 0.9720882, -0.09397608, 0, -0.7518651, -0.5428057, 0.374247, 0, 0.5237069, 0.8516377, -0.02107818, 0, 0.6333505, 0.1926167, -0.7495105, 0, -0.06788242, 0.3998306, 0.9140719, 0, -0.5538629, -0.4729897, -0.6852129, 0, -0.7261456, -0.5911991, 0.3509933, 0, -0.9229275, -0.1782809, 0.3412049, 0, -0.6968815, 0.6511275, 0.300648, 0, 0.9608045, -0.2098363, -0.1811725, 0, 0.06817146, -0.9743405, 0.2145069, 0, -0.3577285, -0.6697087, -0.6507846, 0, -0.1868621, 0.7648617, -0.6164975, 0, -0.6541697, 0.3967915, 0.6439087, 0, 0.699334, -0.6164538, 0.3618239, 0, -0.1546666, 0.6291284, 0.7617583, 0, -0.6841613, -0.2580482, -0.6821542, 0, 0.5383981, 0.4258655, 0.727163, 0, -0.5026988, -0.7939833, -0.3418837, 0, 0.3202972, 0.2834415, 0.9039196, 0, 0.8683227, -0.0003762656, -0.4959995, 0, 0.7911201, -0.08511046, 0.6057106, 0, -0.04011016, -0.4397249, 0.8972364, 0, 0.914512, 0.3579346, -0.1885488, 0, -0.9612039, -0.2756484, 0.01024667, 0, 0.6510361, -0.2877799, -0.7023779, 0, -0.2041786, 0.7365237, 0.6448596, 0, -0.7718264, 0.3790627, 0.5104856, 0, -0.3060083, -0.7692988, 0.5608371, 0, 0.4540073, -0.5024843, 0.73579, 0, 0.4816796, 0.6021208, -0.636738, 0, 0.696198, -0.3222197, 0.6414692, 0, -0.6532161, -0.6781149, 0.3368516, 0, 0.5089301, -0.6154662, -0.6018234, 0, -0.163592, -0.9133605, -0.3728409, 0, 0.5240802, -0.8437664, 0.1157506, 0, 0.5902587, 0.4983818, -0.6349884, 0, 0.5863228, 0.4947647, 0.6414308, 0, 0.6779335, 0.2341345, 0.6968409, 0, 0.7177054, -0.6858979, 0.1201786, 0, -0.532882, -0.5205125, 0.6671608, 0, -0.8654874, -0.07007271, -0.4960054, 0, -0.286181, 0.7952089, 0.5345495, 0, -0.0484953, 0.9810836, -0.1874116, 0, -0.6358522, 0.6058348, 0.47818, 0, 0.6254795, -0.286162, 0.7258697, 0, -0.258526, 0.5061949, -0.8227582, 0, 0.02136307, 0.5064017, -0.862033, 0, 0.2001118, 0.8599263, 0.4695551, 0, 0.4743561, 0.6014985, -0.6427953, 0, 0.6622994, -0.5202475, -0.539168, 0, 0.08084973, -0.653272, 0.7527941, 0, -0.6893687, 0.05928604, 0.7219805, 0, -0.1121887, -0.9673185, 0.2273953, 0, 0.7344116, 0.5979668, -0.3210533, 0, 0.5789393, -0.248885, 0.776457, 0, 0.6988183, 0.355717, -0.6205791, 0, -0.8636845, -0.2748771, -0.4224826, 0, -0.4247028, -0.4640881, 0.777335, 0, 0.5257723, -0.8427017, 0.115833, 0, 0.934383, 0.3163025, -0.1639544, 0, -0.1016836, -0.8057303, -0.5834888, 0, -0.6529239, 0.5060213, -0.5635893, 0, -0.2465286, -0.9668206, -0.06694497, 0, -0.9776897, -0.2099251, -0.007368825, 0, 0.7736893, 0.5734245, 0.2694238, 0, -0.6095088, 0.4995679, 0.6155737, 0, 0.5794535, 0.7434547, 0.3339292, 0, -0.8226211, 0.08142582, 0.5627294, 0, -0.5103855, 0.4703668, 0.719904, 0, -0.5764972, -0.07231656, -0.8138927, 0, 0.7250629, 0.3949971, -0.5641463, 0, -0.1525424, 0.4860841, -0.8604958, 0, -0.5550976, -0.4957821, 0.6678823, 0, -0.1883614, 0.914587, 0.3578417, 0, 0.7625557, -0.5414408, -0.354049, 0, -0.5870232, -0.3226498, -0.7424964, 0, 0.3051124, 0.2262544, -0.9250488, 0, 0.6379576, 0.5772424, -0.509707, 0, -0.5966776, 0.1454852, -0.7891831, 0, -0.6583306, 0.6555488, -0.3699415, 0, 0.7434893, 0.2351085, 0.6260573, 0, 0.5562114, 0.826436, -0.08736329, 0, -0.302894, -0.8251527, 0.4768419, 0, 0.1129344, -0.9858884, -0.1235711, 0, 0.5937653, -0.5896814, 0.5474657, 0, 0.6757964, -0.5835758, -0.4502648, 0, 0.7242303, -0.115272, 0.679855, 0, -0.9511914, 0.0753624, -0.2992581, 0, 0.2539471, -0.1886339, 0.9486454, 0, 0.5714336, -0.1679451, -0.8032796, 0, -0.06778235, 0.3978269, 0.9149532, 0, 0.6074973, 0.73306, -0.3058923, 0, -0.5435479, 0.1675822, 0.8224791, 0, -0.5876678, -0.3380045, -0.7351187, 0, -0.7967563, 0.04097823, -0.6029099, 0, -0.1996351, 0.8706295, 0.4496111, 0, -0.0278766, -0.9106233, -0.4122962, 0, -0.7797626, -0.6257635, 0.01975776, 0, -0.5211233, 0.7401645, -0.4249555, 0, 0.8575425, 0.4053273, -0.3167502, 0, 0.1045223, 0.8390196, -0.5339674, 0, 0.3501823, 0.9242524, -0.152085, 0, 0.198785, 0.07647613, 0.9770547, 0, 0.7845997, 0.6066257, -0.1280964, 0, 0.09006737, -0.975099, -0.2026569, 0, -0.8274344, -0.5422996, 0.1458204, 0, -0.3485798, -0.4158023, 0.8400004, 0, -0.2471779, -0.730482, -0.6366311, 0, -0.3700155, 0.8577948, 0.3567584, 0, 0.5913395, -0.5483119, -0.5913303, 0, 0.1204874, -0.7626472, -0.6354935, 0, 0.6169593, 0.03079648, 0.7863923, 0, 0.1258157, -0.664083, -0.7369968, 0, -0.6477565, -0.1740147, -0.7417077, 0, 0.6217889, -0.7804431, -0.06547655, 0, 0.6589943, -0.6096988, 0.4404474, 0, -0.2689838, -0.6732403, -0.6887636, 0, -0.3849775, 0.5676543, 0.7277094, 0, 0.5754445, 0.8110471, -0.1051963, 0, 0.9141594, 0.3832948, 0.1319006, 0, -0.1079253, 0.9245494, 0.3654594, 0, 0.3779771, 0.3043149, 0.8743716, 0, -0.2142885, -0.8259286, 0.5214617, 0, 0.5802544, 0.4148099, -0.7008834, 0, -0.1982661, 0.8567162, -0.4761597, 0, -0.03381554, 0.3773181, -0.9254661, 0, -0.6867923, -0.6656598, 0.2919134, 0, 0.7731743, -0.2875794, -0.565243, 0, -0.09655942, 0.9193708, -0.3813575, 0, 0.2715702, -0.957791, -0.09426606, 0, 0.2451016, -0.6917999, -0.6792188, 0, 0.9777008, -0.1753855, 0.1155037, 0, -0.522474, 0.8521607, 0.02903616, 0, -0.773488, -0.5261292, 0.353418, 0, -0.7134492, -0.2695473, 0.6467878, 0, 0.1644037, 0.5105846, -0.8439637, 0, 0.6494636, 0.05585611, 0.7583384, 0, -0.4711971, 0.5017281, -0.7254256, 0, -0.6335765, -0.2381686, -0.7361091, 0, -0.9021533, -0.2709478, -0.3357182, 0, -0.3793711, 0.8722581, 0.3086152, 0, -0.6855599, -0.3250143, 0.6514394, 0, 0.2900942, -0.7799058, -0.5546101, 0, -0.2098319, 0.8503707, 0.4825352, 0, -0.4592604, 0.6598504, -0.5947077, 0, 0.8715945, 0.09616365, -0.4807031, 0, -0.6776666, 0.7118505, -0.1844907, 0, 0.7044378, 0.3124276, 0.637304, 0, -0.7052319, -0.2401093, -0.6670798, 0, 0.081921, -0.7207336, -0.6883546, 0, -0.6993681, -0.5875763, -0.4069869, 0, -0.1281454, 0.6419896, 0.7559286, 0, -0.6337388, -0.6785471, -0.3714147, 0, 0.5565052, -0.2168888, -0.8020357, 0, -0.5791554, 0.7244372, -0.3738579, 0, 0.1175779, -0.7096451, 0.6946793, 0, -0.613462, 0.1323631, 0.7785528, 0, 0.6984636, -0.02980516, -0.7150247, 0, 0.8318083, -0.3930172, 0.3919598, 0, 0.1469576, 0.05541652, -0.9875892, 0, 0.7088686, -0.2690504, 0.6520101, 0, 0.2726053, 0.6736977, -0.6868899, 0, -0.6591296, 0.3035459, -0.6880466, 0, 0.4815131, -0.752827, 0.4487723, 0, 0.943001, 0.1675647, -0.2875261, 0, 0.4348029, 0.7695305, -0.4677278, 0, 0.3931996, 0.5944736, 0.7014236, 0, 0.7254336, -0.6039256, 0.3301815, 0, 0.7590235, -0.6506083, 0.02433313, 0, -0.8552769, -0.3430043, 0.3883936, 0, -0.6139747, 0.6981725, 0.3682258, 0, -0.7465906, -0.575201, 0.3342849, 0, 0.5730066, 0.8105555, -0.1210917, 0, -0.9225878, -0.3475211, -0.167514, 0, -0.7105817, -0.4719692, -0.5218417, 0, -0.0856461, 0.3583001, 0.9296697, 0, -0.8279698, -0.2043157, 0.5222271, 0, 0.427944, 0.278166, 0.8599346, 0, 0.539908, -0.7857121, -0.3019204, 0, 0.5678404, -0.5495414, -0.6128308, 0, -0.9896071, 0.1365639, -0.04503419, 0, -0.6154343, -0.6440876, 0.4543037, 0, 0.1074204, -0.794634, 0.5975094, 0, -0.359545, -0.888553, 0.2849578, 0, -0.2180405, 0.1529889, 0.9638738, 0, -0.7277432, -0.6164051, -0.3007235, 0, 0.7249729, -0.006697195, 0.6887448, 0, -0.5553659, -0.5336586, 0.6377908, 0, 0.5137558, 0.7976208, -0.316, 0, -0.3794025, 0.9245608, -0.03522751, 0, 0.8229249, 0.2745366, -0.4974177, 0, -0.5404114, 0.6091142, 0.5804614, 0, 0.8036582, -0.270303, 0.5301602, 0, 0.6044319, 0.6832969, 0.4095943, 0, 0.06389989, 0.9658208, -0.2512108, 0, 0.1087113, 0.7402471, -0.6634878, 0, -0.7134277, -0.6926784, 0.1059128, 0, 0.6458898, -0.5724549, -0.5050958, 0, -0.6553931, 0.7381471, 0.1599956, 0, 0.3910961, 0.9188871, -0.05186756, 0, -0.4879023, -0.5904377, 0.6429111, 0, 0.601479, 0.7707441, -0.210182, 0, -0.5677173, 0.7511361, 0.3368852, 0, 0.7858574, 0.2266747, 0.5753667, 0, -0.4520346, -0.6042227, -0.6561857, 0, 0.002272116, 0.4132844, -0.9105992, 0, -0.5815752, -0.5162926, 0.6286591, 0, -0.03703705, 0.8273786, 0.5604221, 0, -0.5119693, 0.7953544, -0.324498, 0, -0.2682417, -0.957229, -0.1084388, 0, -0.2322483, -0.9679131, -0.09594243, 0, 0.3554329, -0.8881506, 0.2913006, 0, 0.734652, -0.4371373, 0.5188423, 0, 0.998512, 0.04659011, -0.02833945, 0, -0.3727688, -0.9082481, 0.1900757, 0, 0.9173738, -0.3483642, 0.1925298, 0, 0.2714911, 0.414753, -0.8684887, 0, 0.5131763, -0.7116334, 0.4798207, 0, -0.8737354, 0.1888699, -0.4482351, 0, 0.8460044, -0.3725218, 0.38145, 0, 0.8978727, -0.1780209, -0.4026575, 0, 0.2178066, -0.9698323, -0.109479, 0, -0.1518031, -0.7788918, -0.6085091, 0, -0.2600385, -0.4755398, -0.840382, 0, 0.5723135, -0.7474341, -0.3373418, 0, -0.7174141, 0.1699017, -0.6756111, 0, -0.6841808, 0.02145708, -0.7289968, 0, -0.2007448, 0.06555606, -0.9774477, 0, -0.1148804, -0.8044887, 0.5827524, 0, -0.787035, 0.03447489, 0.6159443, 0, -0.2015596, 0.6859872, 0.6991389, 0, -0.08581083, -0.1092084, -0.990308, 0, 0.5532693, 0.7325251, -0.3966108, 0, -0.1842489, -0.9777375, -0.1004077, 0, 0.07754738, -0.9111506, 0.404711, 0, 0.1399838, 0.7601631, -0.6344734, 0, 0.4484419, -0.8452892, 0.2904925, 0);

const PRIME_X: i32 = 501125321;
const PRIME_Y: i32 = 1136930381;
const PRIME_Z: i32 = 1720413743;

fn fnl_fast_floor(f: f32) -> i32 { return i32(floor(f)); }
fn fnl_fast_round(f: f32) -> i32 { return i32(round(f)); }
fn fnl_lerp(a: f32, b: f32, t: f32) -> f32 { return mix(a, b, t); }
fn fnl_fast_abs(f: f32) -> f32 { return abs(f); }
fn fnl_interp_hermite(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }

fn fnl_calc_fractal_bounding(octaves: i32, gain: f32) -> f32 {
    var g: f32 = abs(gain);
    var amp: f32 = g;
    var amp_fractal: f32 = 1.0;
    for (var i: i32 = 1; i < octaves; i = i + 1) {
        amp_fractal = amp_fractal + amp;
        amp = amp * g;
    }
    return 1.0 / amp_fractal;
}

fn fnl_hash3d(seed: i32, x_primed: i32, y_primed: i32, z_primed: i32) -> i32 {
    var hash: i32 = seed ^ x_primed ^ y_primed ^ z_primed;
    hash = hash * 668265261;
    return hash;
}

fn fnl_val_coord3d(seed: i32, x_primed: i32, y_primed: i32, z_primed: i32) -> f32 {
    var hash: i32 = fnl_hash3d(seed, x_primed, y_primed, z_primed);
    hash = hash * hash;
    hash = hash ^ (hash << 19);
    return f32(hash) * (1.0 / 2147483648.0);
}

fn fnl_grad_coord3d(seed: i32, x_primed: i32, y_primed: i32, z_primed: i32, xd: f32, yd: f32, zd: f32) -> f32 {
    var hash: i32 = fnl_hash3d(seed, x_primed, y_primed, z_primed);
    hash = hash ^ (hash >> 15u);
    hash = hash & 252;
    return xd * GRADIENTS_3D[hash] + yd * GRADIENTS_3D[hash | 1] + zd * GRADIENTS_3D[hash | 2];
}

fn fnl_grad_coord_out3d(seed: i32, x_primed: i32, y_primed: i32, z_primed: i32) -> vec3<f32> {
    let hash: i32 = fnl_hash3d(seed, x_primed, y_primed, z_primed) & 1020;
    return vec3<f32>(RAND_VECS_3D[hash], RAND_VECS_3D[hash | 1], RAND_VECS_3D[hash | 2]);
}

fn fnl_grad_coord_dual3d(seed: i32, x_primed: i32, y_primed: i32, z_primed: i32, xd: f32, yd: f32, zd: f32) -> vec3<f32> {
    let hash: i32 = fnl_hash3d(seed, x_primed, y_primed, z_primed);
    let index1: i32 = hash & 252;
    let index2: i32 = (hash >> 6u) & 1020;
    let xg: f32 = GRADIENTS_3D[index1];
    let yg: f32 = GRADIENTS_3D[index1 | 1];
    let zg: f32 = GRADIENTS_3D[index1 | 2];
    let value: f32 = xd * xg + yd * yg + zd * zg;
    let xgo: f32 = RAND_VECS_3D[index2];
    let ygo: f32 = RAND_VECS_3D[index2 | 1];
    let zgo: f32 = RAND_VECS_3D[index2 | 2];
    return vec3<f32>(value * xgo, value * ygo, value * zgo);
}

fn fnl_single_opensimplex2_3d(seed_in: i32, x_in: f32, y_in: f32, z_in: f32) -> f32 {
    var seed = seed_in;
    var x = x_in;
    var y = y_in;
    var z = z_in;

    var i: i32 = fnl_fast_round(x);
    var j: i32 = fnl_fast_round(y);
    var k: i32 = fnl_fast_round(z);
    var x0: f32 = x - f32(i);
    var y0: f32 = y - f32(j);
    var z0: f32 = z - f32(k);

    var x_n_sign: i32 = i32(-1.0 - x0) | 1;
    var y_n_sign: i32 = i32(-1.0 - y0) | 1;
    var z_n_sign: i32 = i32(-1.0 - z0) | 1;

    var ax0: f32 = f32(x_n_sign) * -x0;
    var ay0: f32 = f32(y_n_sign) * -y0;
    var az0: f32 = f32(z_n_sign) * -z0;

    i = i * PRIME_X;
    j = j * PRIME_Y;
    k = k * PRIME_Z;

    var value: f32 = 0.0;
    var a: f32 = (0.6 - x0 * x0) - (y0 * y0 + z0 * z0);

    for (var l: i32 = 0; ; l = l + 1) {
        if (a > 0.0) {
            value = value + (a * a) * (a * a) * fnl_grad_coord3d(seed, i, j, k, x0, y0, z0);
        }

        var b: f32 = a + 1.0;
        var i1: i32 = i;
        var j1: i32 = j;
        var k1: i32 = k;
        var x1: f32 = x0;
        var y1: f32 = y0;
        var z1: f32 = z0;
        if (ax0 >= ay0 && ax0 >= az0) {
            x1 = x1 + f32(x_n_sign);
            b = b - f32(x_n_sign) * 2.0 * x1;
            i1 = i1 - x_n_sign * PRIME_X;
        } else if (ay0 > ax0 && ay0 >= az0) {
            y1 = y1 + f32(y_n_sign);
            b = b - f32(y_n_sign) * 2.0 * y1;
            j1 = j1 - y_n_sign * PRIME_Y;
        } else {
            z1 = z1 + f32(z_n_sign);
            b = b - f32(z_n_sign) * 2.0 * z1;
            k1 = k1 - z_n_sign * PRIME_Z;
        }

        if (b > 0.0) {
            value = value + (b * b) * (b * b) * fnl_grad_coord3d(seed, i1, j1, k1, x1, y1, z1);
        }

        if (l == 1) { break; }

        ax0 = 0.5 - ax0;
        ay0 = 0.5 - ay0;
        az0 = 0.5 - az0;

        x0 = f32(x_n_sign) * ax0;
        y0 = f32(y_n_sign) * ay0;
        z0 = f32(z_n_sign) * az0;

        a = a + (0.75 - ax0) - (ay0 + az0);

        i = i + ((x_n_sign >> 1u) & PRIME_X);
        j = j + ((y_n_sign >> 1u) & PRIME_Y);
        k = k + ((z_n_sign >> 1u) & PRIME_Z);

        x_n_sign = -x_n_sign;
        y_n_sign = -y_n_sign;
        z_n_sign = -z_n_sign;

        seed = ~seed;
    }

    return value * 32.69428253173828125;
}

fn fnl_single_value_3d(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let x0: i32 = fnl_fast_floor(x);
    let y0: i32 = fnl_fast_floor(y);
    let z0: i32 = fnl_fast_floor(z);

    let xs: f32 = fnl_interp_hermite(x - f32(x0));
    let ys: f32 = fnl_interp_hermite(y - f32(y0));
    let zs: f32 = fnl_interp_hermite(z - f32(z0));

    var xp0: i32 = x0 * PRIME_X;
    var yp0: i32 = y0 * PRIME_Y;
    var zp0: i32 = z0 * PRIME_Z;
    let xp1: i32 = xp0 + PRIME_X;
    let yp1: i32 = yp0 + PRIME_Y;
    let zp1: i32 = zp0 + PRIME_Z;

    let xf00: f32 = fnl_lerp(fnl_val_coord3d(seed, xp0, yp0, zp0), fnl_val_coord3d(seed, xp1, yp0, zp0), xs);
    let xf10: f32 = fnl_lerp(fnl_val_coord3d(seed, xp0, yp1, zp0), fnl_val_coord3d(seed, xp1, yp1, zp0), xs);
    let xf01: f32 = fnl_lerp(fnl_val_coord3d(seed, xp0, yp0, zp1), fnl_val_coord3d(seed, xp1, yp0, zp1), xs);
    let xf11: f32 = fnl_lerp(fnl_val_coord3d(seed, xp0, yp1, zp1), fnl_val_coord3d(seed, xp1, yp1, zp1), xs);

    let yf0: f32 = fnl_lerp(xf00, xf10, ys);
    let yf1: f32 = fnl_lerp(xf01, xf11, ys);

    return fnl_lerp(yf0, yf1, zs);
}

fn fnl_transform_noise_coord_opensimplex2(x: ptr<function, f32>, y: ptr<function, f32>, z: ptr<function, f32>, freq: f32) {
    *x = *x * freq;
    *y = *y * freq;
    *z = *z * freq;
    let R3: f32 = 2.0 / 3.0;
    let r: f32 = (*x + *y + *z) * R3;
    *x = r - *x;
    *y = r - *y;
    *z = r - *z;
}

fn fnl_transform_noise_coord_value(x: ptr<function, f32>, y: ptr<function, f32>, z: ptr<function, f32>, freq: f32) {
    *x = *x * freq;
    *y = *y * freq;
    *z = *z * freq;
}

fn fnl_transform_warp_coord(x: ptr<function, f32>, y: ptr<function, f32>, z: ptr<function, f32>) {
    let R3: f32 = 2.0 / 3.0;
    let r: f32 = (*x + *y + *z) * R3;
    *x = r - *x;
    *y = r - *y;
    *z = r - *z;
}

fn fnl_single_domain_warp_opensimplex2_gradient(seed_in: i32, warp_amp: f32, frequency: f32, x_in: f32, y_in: f32, z_in: f32) -> vec3<f32> {
    var seed = seed_in;
    var x = x_in;
    var y = y_in;
    var z = z_in;
    x = x * frequency;
    y = y * frequency;
    z = z * frequency;

    var i: i32 = fnl_fast_round(x);
    var j: i32 = fnl_fast_round(y);
    var k: i32 = fnl_fast_round(z);
    var x0: f32 = x - f32(i);
    var y0: f32 = y - f32(j);
    var z0: f32 = z - f32(k);

    var x_n_sign: i32 = i32(-x0 - 1.0) | 1;
    var y_n_sign: i32 = i32(-y0 - 1.0) | 1;
    var z_n_sign: i32 = i32(-z0 - 1.0) | 1;

    var ax0: f32 = f32(x_n_sign) * -x0;
    var ay0: f32 = f32(y_n_sign) * -y0;
    var az0: f32 = f32(z_n_sign) * -z0;

    i = i * PRIME_X;
    j = j * PRIME_Y;
    k = k * PRIME_Z;

    var vx: f32 = 0.0;
    var vy: f32 = 0.0;
    var vz: f32 = 0.0;

    var a: f32 = (0.6 - x0 * x0) - (y0 * y0 + z0 * z0);
    for (var l: i32 = 0; l < 2; l = l + 1) {
        if (a > 0.0) {
            let aaaa: f32 = (a * a) * (a * a);
            let g: vec3<f32> = fnl_grad_coord_dual3d(seed, i, j, k, x0, y0, z0);
            vx = vx + aaaa * g.x;
            vy = vy + aaaa * g.y;
            vz = vz + aaaa * g.z;
        }

        var b: f32 = a + 1.0;
        var i1: i32 = i;
        var j1: i32 = j;
        var k1: i32 = k;
        var x1: f32 = x0;
        var y1: f32 = y0;
        var z1: f32 = z0;
        if (ax0 >= ay0 && ax0 >= az0) {
            x1 = x1 + f32(x_n_sign);
            b = b - f32(x_n_sign) * 2.0 * x1;
            i1 = i1 - x_n_sign * PRIME_X;
        } else if (ay0 > ax0 && ay0 >= az0) {
            y1 = y1 + f32(y_n_sign);
            b = b - f32(y_n_sign) * 2.0 * y1;
            j1 = j1 - y_n_sign * PRIME_Y;
        } else {
            z1 = z1 + f32(z_n_sign);
            b = b - f32(z_n_sign) * 2.0 * z1;
            k1 = k1 - z_n_sign * PRIME_Z;
        }

        if (b > 0.0) {
            let bbbb: f32 = (b * b) * (b * b);
            let g: vec3<f32> = fnl_grad_coord_dual3d(seed, i1, j1, k1, x1, y1, z1);
            vx = vx + bbbb * g.x;
            vy = vy + bbbb * g.y;
            vz = vz + bbbb * g.z;
        }

        if (l == 1) { break; }

        ax0 = 0.5 - ax0;
        ay0 = 0.5 - ay0;
        az0 = 0.5 - az0;

        x0 = f32(x_n_sign) * ax0;
        y0 = f32(y_n_sign) * ay0;
        z0 = f32(z_n_sign) * az0;

        a = a + (0.75 - ax0) - (ay0 + az0);

        i = i + ((x_n_sign >> 1u) & PRIME_X);
        j = j + ((y_n_sign >> 1u) & PRIME_Y);
        k = k + ((z_n_sign >> 1u) & PRIME_Z);

        x_n_sign = -x_n_sign;
        y_n_sign = -y_n_sign;
        z_n_sign = -z_n_sign;

        seed = seed + 1293373;
    }

    return vec3<f32>(vx * warp_amp, vy * warp_amp, vz * warp_amp);
}

fn fnl_domain_warp_3d(dir: vec3<f32>, seed: i32, warp_freq: f32, warp_amp: f32) -> vec3<f32> {
    let amp: f32 = warp_amp * fnl_calc_fractal_bounding(3, 0.5);
    var xs: f32 = dir.x;
    var ys: f32 = dir.y;
    var zs: f32 = dir.z;
    fnl_transform_warp_coord(&xs, &ys, &zs);
    let offset: vec3<f32> = fnl_single_domain_warp_opensimplex2_gradient(seed, amp * 32.69428253173828125, warp_freq, xs, ys, zs);
    return dir + offset;
}

fn fnl_fbm_opensimplex2_3d(seed: i32, pos: vec3<f32>, freq: f32, octaves: i32, lac: f32, gain: f32) -> f32 {
    var x: f32 = pos.x;
    var y: f32 = pos.y;
    var z: f32 = pos.z;
    fnl_transform_noise_coord_opensimplex2(&x, &y, &z, freq);

    var s: i32 = seed;
    var sum: f32 = 0.0;
    var amp: f32 = fnl_calc_fractal_bounding(octaves, gain);

    for (var i: i32 = 0; i < octaves; i = i + 1) {
        let noise: f32 = fnl_single_opensimplex2_3d(s, x, y, z);
        sum = sum + noise * amp;
        amp = amp * fnl_lerp(1.0, (noise + 1.0) * 0.5, 0.0);
        x = x * lac;
        y = y * lac;
        z = z * lac;
        amp = amp * gain;
        s = s + 1;
    }
    return sum;
}

fn fnl_ridged_opensimplex2_3d(seed: i32, pos: vec3<f32>, freq: f32, octaves: i32, lac: f32, gain: f32) -> f32 {
    var x: f32 = pos.x;
    var y: f32 = pos.y;
    var z: f32 = pos.z;
    fnl_transform_noise_coord_opensimplex2(&x, &y, &z, freq);

    var s: i32 = seed;
    var sum: f32 = 0.0;
    var amp: f32 = fnl_calc_fractal_bounding(octaves, gain);

    for (var i: i32 = 0; i < octaves; i = i + 1) {
        let noise: f32 = abs(fnl_single_opensimplex2_3d(s, x, y, z));
        sum = sum + (noise * -2.0 + 1.0) * amp;
        amp = amp * fnl_lerp(1.0, 1.0 - noise, 0.0);
        x = x * lac;
        y = y * lac;
        z = z * lac;
        amp = amp * gain;
        s = s + 1;
    }
    return sum;
}

fn fnl_fbm_value_3d(seed: i32, pos: vec3<f32>, freq: f32, octaves: i32, lac: f32, gain: f32) -> f32 {
    var x: f32 = pos.x;
    var y: f32 = pos.y;
    var z: f32 = pos.z;
    fnl_transform_noise_coord_value(&x, &y, &z, freq);

    var s: i32 = seed;
    var sum: f32 = 0.0;
    var amp: f32 = fnl_calc_fractal_bounding(octaves, gain);

    for (var i: i32 = 0; i < octaves; i = i + 1) {
        let noise: f32 = fnl_single_value_3d(s, x, y, z);
        sum = sum + noise * amp;
        amp = amp * fnl_lerp(1.0, (noise + 1.0) * 0.5, 0.0);
        x = x * lac;
        y = y * lac;
        z = z * lac;
        amp = amp * gain;
        s = s + 1;
    }
    return sum;
}

struct ElevationParams {
    seed: i32,
    sea_level: f32,
    continental_freq: f32,
    continental_amp: f32,
    continental_octaves: i32,
    mountain_freq: f32,
    mountain_amp: f32,
    mountain_octaves: i32,
    hill_freq: f32,
    hill_amp: f32,
    hill_octaves: i32,
    detail_freq: f32,
    detail_amp: f32,
    detail_octaves: i32,
    warp_freq: f32,
    warp_amp: f32,
    lacunarity: f32,
    gain: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: ElevationParams;
@group(0) @binding(1) var<storage, read> dirs: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> elevs: array<f32>;

fn compute_elevation(dir: vec3<f32>, p: ElevationParams) -> f32 {
    let warped: vec3<f32> = fnl_domain_warp_3d(dir, p.seed, p.warp_freq, p.warp_amp);

    let continental: f32 = fnl_fbm_opensimplex2_3d(p.seed, warped, p.continental_freq, p.continental_octaves, p.lacunarity, p.gain);

    let mountain_raw: f32 = fnl_ridged_opensimplex2_3d(p.seed, warped, p.mountain_freq, p.mountain_octaves, p.lacunarity, p.gain);
    let mountain_mask: f32 = max(0.0, continental);
    let mountains: f32 = mountain_raw * mountain_mask;

    let hills: f32 = fnl_fbm_opensimplex2_3d(p.seed, warped, p.hill_freq, p.hill_octaves, p.lacunarity, p.gain);

    let detail: f32 = fnl_fbm_value_3d(p.seed, warped, p.detail_freq, p.detail_octaves, p.lacunarity, p.gain);

    return continental * p.continental_amp
         + mountains * p.mountain_amp
         + hills * p.hill_amp
         + detail * p.detail_amp;
}

// Smoothed elevation for normal computation — excludes high-frequency detail
// (Value noise, axis-aligned) to avoid grid-aligned artifacts in normals.
fn compute_elevation_for_normals(dir: vec3<f32>, p: ElevationParams) -> f32 {
    let warped: vec3<f32> = fnl_domain_warp_3d(dir, p.seed, p.warp_freq, p.warp_amp);

    let continental: f32 = fnl_fbm_opensimplex2_3d(p.seed, warped, p.continental_freq, p.continental_octaves, p.lacunarity, p.gain);

    let mountain_raw: f32 = fnl_ridged_opensimplex2_3d(p.seed, warped, p.mountain_freq, p.mountain_octaves, p.lacunarity, p.gain);
    let mountain_mask: f32 = max(0.0, continental);
    let mountains: f32 = mountain_raw * mountain_mask;

    let hills: f32 = fnl_fbm_opensimplex2_3d(p.seed, warped, p.hill_freq, p.hill_octaves, p.lacunarity, p.gain);

    return continental * p.continental_amp
         + mountains * p.mountain_amp
         + hills * p.hill_amp;
}

@compute @workgroup_size(64)
fn elevation_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= arrayLength(&dirs)) {
        return;
    }
    let dir: vec3<f32> = dirs[i];
    elevs[i] = compute_elevation(dir, params);
}

