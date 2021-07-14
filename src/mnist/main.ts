import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { SymbolicTensor } from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'

const trainImage: string = require("../../data/mnist/train-images.idx3-ubyte")
const trainLabel: string = require("../../data/mnist/train-labels.idx1-ubyte")
const testImage: string = require("../../data/mnist/t10k-images.idx3-ubyte")
const testLabel: string = require("../../data/mnist/t10k-labels.idx1-ubyte")

const canvas = document.querySelector<HTMLCanvasElement>('#canvas')
const ctx = canvas.getContext('2d')

function parseImageData(d: ArrayBuffer){
    const dv = new DataView(d)

    const size = dv.getInt32(4, false)
    const nrows = dv.getInt32(8, false)
    const ncols = dv.getInt32(12, false)

    const tensor = tf.tensor(new Uint8Array(d.slice(16)))
    return tensor.reshape([size, nrows, ncols, 1])
}

function parseLabelData(d: ArrayBuffer){
    const dv = new DataView(d)
    const tensor = tf.tensor(new Uint8Array(d.slice(8)))
    return tf.oneHot(tensor, 10)
}

function drawNumber(n: Uint8Array){
    const id = ctx.createImageData(ctx.canvas.width, ctx.canvas.height)
    for(let i = 0; i < n.length; i ++){
        id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = n[i]
        id.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(id, 0, 0)
}

async function getDataset(){
    
    const tX = await fetch(trainImage).then(r => r.arrayBuffer()).then(parseImageData)
    const tY = await fetch(trainLabel).then(r => r.arrayBuffer()).then(parseLabelData)    
    const tsX = await fetch(testImage).then(r => r.arrayBuffer()).then(parseImageData)
    const tsY = await fetch(testLabel).then(r => r.arrayBuffer()).then(parseLabelData)    

    return { tX, tY, tsX, tsY }
}

async function initModel() {

    const model = tf.sequential({
        layers: [
            tf.layers.conv2d({
                inputShape: [28, 28, 1],
                filters: 8,
                kernelSize: 5,
                strides: 1,
                activation: 'relu',
                kernelInitializer: 'varianceScaling'
            }),
            tf.layers.averagePooling2d({
                poolSize: [2, 2],
                strides: [2, 2]
            }),
            tf.layers.conv2d({
                kernelSize: 5,
                filters: 16,
                strides: 1,
                activation: 'relu',
                kernelInitializer: 'varianceScaling'
            }),
            tf.layers.averagePooling2d({
                poolSize: [2, 2],
                strides: [2, 2]
            }),
            tf.layers.flatten(),
            tf.layers.dense({
                units: 128,
                activation: 'relu'
            }),
            tf.layers.dense({
                units: 10,
                activation: 'softmax'
            })
        ]
    })

    const optimizer = tf.train.adam();
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })
    tfvis.show.modelSummary({ name:'summary', tab:'Model' }, model);

    return {
        model
    }
}

function initVis(){
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Train', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    return fitCallbacks
}

async function fit(model: tf.Sequential, tX: tf.Tensor, tY: tf.Tensor, tsX: tf.Tensor, tsY: tf.Tensor, fitCallbacks: tf.CustomCallbackArgs){

    const history = []

    await model.fit(tX, tY, {
        batchSize: 256,
        validationData: [tsX, tsY],
        epochs: 1,
        shuffle: true,
        callbacks: {
            onBatchEnd(batch, logs){
                history.push(logs)
                fitCallbacks.onBatchEnd?.(batch, logs)
            },
            onEpochBegin(epoch){
                console.log(`Epoch: ${epoch} 시작`)
            },
            onEpochEnd(epoch, logs){
                console.log(epoch, logs)
                fitCallbacks.onEpochEnd?.(epoch, logs)
            }
        }
    });
    return history
}

function initCanvas(){
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height)

    let isClicked = false
    canvas.onmousedown = () => isClicked = true
    canvas.onmouseup = () => isClicked = false
    canvas.onmouseleave = () => isClicked = false

    canvas.onmousemove = (e) => {
        if(isClicked){
            const x = Math.floor(e.offsetX / 20)
            const y = Math.floor(e.offsetY / 20)
            ctx.fillStyle = 'white'
            ctx.fillRect(x, y, 2, 2)
        }
    }

    const btn = document.createElement('button')
    btn.innerHTML = '캔버스 지우기'
    btn.onclick = function(){
        ctx.fillStyle = 'black'
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    }
    document.body.append(btn)
}

function getTensorFromCanvas(){
    const id = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
    const buf = new ArrayBuffer(28 * 28)
    const u8 = new Uint8Array(buf)
    for(let i = 0; i < ctx.canvas.width * ctx.canvas.height; i++){
        u8[i] = id.data[i * 4]
    }

    const t = tf.tensor(u8)
    return t.reshape([1, 28, 28, 1])
}

initCanvas()

function displayInferenceResult(p: number[]){
    const result = document.querySelector('#result')
    result.innerHTML = p.map((el, idx) => `<li>${idx} : ${el.toFixed(2)}</li>`).join('')
}

async function main(){
    console.log('Get dataset')
    const { tX, tY, tsX, tsY } = await getDataset()
    console.log('Init model')
    const { model } = await initModel()
    console.log('Init vis')
    const fitCallbacks = initVis()

    const btn = document.createElement('button')
    btn.innerHTML = '훈련!!'
    let isFitting = false
    btn.onclick = async function(){
        if(isFitting == false) {
            isFitting = true
            await fit(model, tX, tY, tsX, tsY, fitCallbacks)
            await model.save('downloads://my-model');
            isFitting = false
        }
    }
    document.body.append(btn)

    const btn2 = document.createElement('button')
    btn2.innerHTML = '추론'
    btn2.onclick = async function(){
        const result = model.predict(getTensorFromCanvas()) as tf.Tensor
        const arr: number[] = result.arraySync()[0]
        displayInferenceResult(arr)
    }
    document.body.append(btn2)
}

main()