const SINGLE_TRANSITION_RENDER_OPTIONS = ["Atom", "Volume", "Superquadric"]
const CLUSTER_COLORS = :glasbey_bw_minc_20_maxl_70_n256

const LEFT_KEY = Keyboard.left
const RIGHT_KEY = Keyboard.right
const UP_KEY = Keyboard.up
const DOWN_KEY = Keyboard.down

const LEFT_DOWN = LEFT_KEY & DOWN_KEY
const RIGHT_DOWN = RIGHT_KEY & DOWN_KEY

const EMBEDDED_SCENE_BACKGROUND = colorant"#F5F5F5"
const EMBEDDED_SCENE_SELECTED = colorant"#8b8680"
const DISTANCE_MATRIX_COLORMAP = to_colormap(:linear_worb_100_25_c53_n256)
const CLUSTER_COLORMAP = to_colormap(CLUSTER_COLORS)
const CLUSTER_CONSENSUS_COLORMAP = to_colormap(:linear_worb_100_25_c53_n256)
