var express = require('express');
var ensureLogIn = require('connect-ensure-login').ensureLoggedIn;
var db = require('../db.cjs');

// var ensureLoggedIn = ensureLogIn();


var router = express.Router();

function fetch_models(req, res, next) {
  db.all('SELECT * FROM models', function(err, rows) {
    if (err) { return next(err); }
    
    var models = rows.map(function(row) {
      return {
        id: row.id,
        owner_id: row.owner_id,
        title: row.title,
        date: row.date,
        stars: row.stars,
        thumb_image_path: row.thumb_image_path,
        path: row.path
      }
    });
    res.locals.models = models;
    next();
  });
}

/* GET home page. */
router.get('/', fetch_models, function(req, res, next) {
  // console.log("index.cjs user: %s", req.user);
  res.render('index', { user: req.user });
});


module.exports = router;
