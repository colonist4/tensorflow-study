const path = require('path');
const HtmlWebpackPlugin = require("html-webpack-plugin");

const folders = ['mnist', 'mnist_yolo']

module.exports = {
    entry: folders.reduce((obj, f) => ({
        ...obj,
        [f]: [`./src/${f}/main.ts`]
    }), {}),
    output: {
        path: path.resolve(__dirname, 'dist/frontend'),
        filename: 'js/[name].js',
        chunkFilename: 'js/[name].bundle.js',
    },
    module:{
        rules:[{
            test: /\.ts$/,
            use: [
                {
                    loader: 'ts-loader',
                    options: {
                        transpileOnly: true
                    }
                }
            ],
            exclude: [/node_modules/]
        }, {
            test: /\.(csv|idx3-ubyte|idx1-ubyte)$/,
            type: 'asset/resource'
        }]
    },
    plugins:[
        ... folders.map(
            f => new HtmlWebpackPlugin({
                template: `./src/${f}/index.html`,
                filename: `${f}.html`,
                chunks: [f]
            })
        )
    ],
    resolve:{
        // import 할때 확장자 생략 가능하게
        extensions:['.js', '.ts'],
        // symlinks: false
    },
    watchOptions:{
        ignored: ['node_modules/**', 'dist/**']
    },
    mode: process.env.NODE_ENV === "production" ? "production" : "development",
    devtool: 'eval-cheap-module-source-map',
};