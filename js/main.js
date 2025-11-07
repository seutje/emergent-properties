import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';
import { Renderer } from './render/Renderer.js';

const appRoot = document.getElementById('app');
if (!appRoot) {
  throw new Error('Renderer root element #app is missing from the DOM.');
}

const renderer = new Renderer(appRoot);
renderer.init();

const gui = new GUI();
gui.title('Emergent Properties');
gui.domElement.style.display = 'none'; // placeholder until controls exist

let last = performance.now();
function loop(timestamp) {
  const delta = (timestamp - last) * 0.001;
  last = timestamp;
  renderer.update(delta);
  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
