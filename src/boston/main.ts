import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { SymbolicTensor } from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'

// const trainDataURL: string = require("../data/train.csv")
const dataURL: string = require("../../data/boston.csv")
let model: tf.LayersModel
let ds: tf.data.Dataset<tf.TensorContainer>

async function main() {

    const csvDataset = tf.data.csv(dataURL)

    const X = csvDataset.map((v) => {
        const copied = Object.assign({}, v)
        delete copied['medv']
        return Object.values(copied)
    })
    const Y = csvDataset.map((v) => {
        return v['medv']
    })
    
    const InputLayer = tf.layers.input({ shape: [13] })
    const H1 = tf.layers.dense({ units: 32 }).apply(InputLayer)
    const OutputLayer = tf.layers.dense({ units: 1 }).apply(H1) as SymbolicTensor[]

    model = tf.model({ inputs: InputLayer, outputs: OutputLayer })
    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam()
    })

    ds = tf.data.zip({ xs: X, ys: Y }).batch(32)

    tfvis.visor()
}
main()

function fit(){
    model.fitDataset(
        ds, 
        {
            epochs: 100,
            callbacks: {
                onEpochEnd(epoch, log){
                    console.log(log)
                    console.log(`[${epoch}] loss: ${log.loss}`)
                }
            }
        }
    ).then(info => {
        console.log(info)
    })
}

const btn = document.createElement('button')
btn.innerHTML = '훈련!!'
btn.onclick = fit
document.body.append(btn)