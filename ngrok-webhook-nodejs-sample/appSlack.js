var express = require('express');
var app = express();
app.use(express.json());
const port = 3000;

app.all('/*', function (req, res) {
   console.log("Headers:"+ JSON.stringify(req.headers, null, 3));
   console.log("Body:"+ JSON.stringify(req.body, null, 3));

   if(req.body.challenge!=null){
      //When you enable Event Subscriptions in Slack, Slack makes a one-time post call to the app
      //sending a challenge field value and expects the app to respond with this value.
      res.type('txt');
      res.send(req.body.challenge);
   }else{
      //For all the rest of the requests the app responds the same message.
      res.json({ message: "Thank you for the message" });
   }
})

app.listen(port, function () {
   console.log(`Example Slack app listening at ${port}`)
})