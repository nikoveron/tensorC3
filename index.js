const tf = require('@tensorflow/tfjs');



//tidy y keeper:::
/* console.log('start', tf.memory().numTensors)

let keeper, chaser, seeker, beater
tf.tidy(() =>
{ keeper = tf.tensor([1,2,3]) 
  chaser = tf.tensor([1,2,3]) 
  seeker = tf.tensor([1,2,3])
  beater = tf.tensor([1,2,3])

console.log('dentro ordenado', tf.memory().numTensors)

tf.keep(keeper)
return chaser 
}) 
console.log('después de ordenar', tf.memory().numTensors)

console.log('end', tf.memory().numTensors)
 
 /
//**/


//modelo de bandas musicales:::

const users = ['Gant', 'Todd', 'Jed', 'Justin']
const bands = ['Nirvana', 'Uñas de nueve pulgadas', 'Backstreet Boys', 'N sincronizacion', 'Club Nocturno', 'apashe', 'PLS']
const features = ['grunge', 'Rock','Industrial', 'Banda de chicos', 'Danza', 'Tecno']

const user_votes = tf.tensor([
    [
        10,9,1,1,8,7,8
    ],
    [
        6,8,2,2,0,10,0
    ],
    [
        0,2,10,9,3,7,0
    ],
    [
        7,4,2,3,6,5,5
    ]
])

const band_feats= tf.tensor([
    [
        1,1,0,0,0,0
    ],
    [
        1,0,1,0,0,0
    ],
    [
        0,0,0,1,1,0
    ],
    [
        0,0,0,1,0,0
    ],
    [
        0,0,1,0,0,1
    ],
    [
        0,0,1,0,0,1
    ],
    [
        1,1,0,0,0,0
    ]
])

const user_feats = tf.matMul(user_votes, band_feats)
console.log("aca imprimimos el tensor con los votos, sin formato:")
console.log("")
user_feats.print()

const top_user_features = tf.topk(user_feats, features.length)

const top_genres = top_user_features.indices.arraySync()

console.log("**")
console.log("Aca le damos formato a los votos:")
console.log("")
users.map((u,i)=>{
const categorías_clasificadas = top_genres[i].map(v => features[v])

console.log(u, categorías_clasificadas) })