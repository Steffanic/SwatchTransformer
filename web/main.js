function changeBodyBg(color) {
  document.body.style.background = color;
}
function RGBToFloat32Array(rgb) {
  // Choose correct separator
  let sep = rgb.indexOf(",") > -1 ? "," : " ";
  // Turn "rgb(r,g,b)" into [r,g,b]
  rgb_vals = rgb.substr(4).split(")")[0].split(sep);
  return Float32Array.from(rgb_vals);
}
function componentToHex(c) {
  if(c < 0) c = 0;
  var hex = c.toString(16);
  return hex.length == 1 ? "0" + hex : hex;
}

function rgbToHex(r, g, b) {
  return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
}
function download(filename, text) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

function nameThePalette(){
  // get the name of the palette from the user
  let palette_name = prompt("Please enter the name of the palette", "My Palette");
  if (palette_name == null || palette_name == "") {
    palette_name = "My Palette";
  }
  return palette_name;
}


const alwan = new Alwan('#color-picker', {
  toggle: true
});
let intervalID = -1;
async function clickSubmit() {
  /*if (intervalID != -1) {
    clearInterval(intervalID);
    intervalID = setInterval(run, 500, inputSrcData);
    return;
  }
  intervalID = setInterval(run, 500, inputSrcData);*/
  await run(inputSrcData);
  setPredictedPallete();
  //cycleBackgroundColors();
}

async function savePalette(){
  // save the hex_codes of the colors in the palette in a text file and download it
  let hex_codes = [];
  for (let i = 0; i < outputSeqColors.length; i++) {
    let r = outputSeqColors[i].substr(4).split(")")[0].split(",")[0];
    let g = outputSeqColors[i].substr(4).split(")")[0].split(",")[1];
    let b = outputSeqColors[i].substr(4).split(")")[0].split(",")[2];
    hex_codes.push(rgbToHex(Math.trunc(r),Math.trunc(g),Math.trunc(b)));
  }
  let text = hex_codes.join("\n");
  let filename = nameThePalette() + ".txt";
  download(filename, text);

}

let inputSrcData = Float32Array.from([0.0, 0.0, 0.0]);
let inputTgtData = Float32Array.from([0.0, 0.0, 0.0]);

let MAX_SEQ_LENGTH = 5;
let outputSeqColors = new Array(MAX_SEQ_LENGTH);
alwan.on('change', function (color) {
  // output: { r: 0, g: 0, b: 0, a: 1}
  inputSrcData = RGBToFloat32Array(`${color.rgb()}`).map(x => x / 255.0);
})

function cycleBackgroundColors(){
  let index = 0;
  if (intervalID != -1) {
    clearInterval(intervalID);
  }
  intervalID = setInterval(function(){
    document.body.style.background = outputSeqColors[index];
    index += 1;
    if (index == outputSeqColors.length) index = 0;
  }, 500);
}

function setPredictedPallete(){
  for (let i = 0; i < outputSeqColors.length; i++) {
    let pallete_color = document.getElementById(`predicted-color-${i+1}`);
    pallete_color.style.backgroundColor = outputSeqColors[i];
    let r = outputSeqColors[i].substr(4).split(")")[0].split(",")[0];
    let g = outputSeqColors[i].substr(4).split(")")[0].split(",")[1];
    let b = outputSeqColors[i].substr(4).split(")")[0].split(",")[2];
    pallete_color.innerHTML = `<span>${rgbToHex(Math.trunc(r),Math.trunc(g),Math.trunc(b))}</span>`;
  }
}


async function run(inputSrcData) {
  try {
    // create a new session and load the AlexNet model.
    const session = await ort.InferenceSession.create('../models/swatchTransformer_fiveColor.onnx');

    // prepare dummy input data
    const dims = [1, 1, 3];
    const size = dims[0] * dims[1] * dims[2];
    const tgt_dims = [1, 5, 3];
    const tgt_size = tgt_dims[0] * tgt_dims[1] * tgt_dims[2];
    // make a random traget array
    const targetArray = new Float32Array(tgt_size);
    for (let i = 0; i < tgt_size; i++) {
      targetArray[i] = Math.random();
    }
    inputTgtData = targetArray;
    // prepare feeds. use model input names as keys.
    const feeds = { input_src: new ort.Tensor('float32', inputSrcData, dims), input_tgt: new ort.Tensor('float32', inputTgtData, tgt_dims) };

    // feed inputs and run
    const results = await session.run(feeds);
    for (let i = 0; i < results.output.data.length; i++) {
      results.output.data[i] = results.output.data[i] * 255.0;
    }
    // then grab three elements at a time and make them a color
    for (let i = 0; i < results.output.data.length; i += 3) {
      let color = `rgb(${results.output.data[i]}, ${results.output.data[i + 1]}, ${results.output.data[i + 2]})`;
      outputSeqColors[i / 3] = color;
    }

   
    document.body.style.background = `rgb(${results.output.data.map(x => x * 255)})`
    outputData = results.output.data.map(x => x * 255.0);
    // lets initialize a Float 32 array with the output data and the inputTgtData concatenated in the second dimension 
    /*const next_tgt_dims = [1, i + 2, 3];
    const next_tgt_size = next_tgt_dims[0] * next_tgt_dims[1] * next_tgt_dims[2];
    nextTargetArray = new Float32Array(next_tgt_size);
    for (let j = 0; j < next_tgt_size; j++) {
      if (j < next_tgt_size / 2) {
        nextTargetArray[j] = outputData[j];
      } else {
        nextTargetArray[j] = inputTgtData[j - next_tgt_size / 2];
      }
    }
    inputTgtData = nextTargetArray;
    console.log(`nextTargetArray: ${nextTargetArray}`);
  
    console.log(outputSeqColors);*/
  } catch (e) {
    console.log(e);
  }
}
run(Float32Array.from([0.0, 0.0, 0.0]));