function iniciarMap(){
    var coord = {lat: -40.37223775632141 ,lng: -72.24283810181745};
    var map = new google.maps.Map(document.getElementById('map'),{
      zoom: 10,
      center: coord
    });
    var marker = new google.maps.Marker({
      position: coord,
      map: map
    });
}

const spawner = require('child_process').spawn;
const data_to_pass_in = 'message';
const python_process = spawner('python', ['./rastreador.py', data_to_pass_in]);
python_process.stdout.on('data',[data] => {
      console.log('Data received from py: ', data.toString());
});

//cargar en HTML el valor del ecosistema
var valor = 155
window.onload = function() {
  //when the document is finished loading, replace everything
  //between the <a ...> </a> tags with the value of "valor"
document.getElementById("myLink").innerHTML=valor;
} 



