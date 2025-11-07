import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/dist/lil-gui.umd.min.js';
import { Renderer } from './render/Renderer.js';

const appRoot = document.getElementById('app');
const renderer = new Renderer(appRoot);
renderer.init();

const gui = new GUI();
gui.title('Emergent Properties');
gui.domElement.style.display = 'none'; // placeholder until controls exist

let last = performance.now();
function loop(timestamp) {
  const delta = (timestamp - last) * 0.001;
  last = timestamp;
  renderer.render(delta);
  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
