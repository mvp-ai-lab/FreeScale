var express = require('express');
var ensureLogIn = require('connect-ensure-login').ensureLoggedIn;
var db = require('../db.cjs');

// var ensureLoggedIn = ensureLogIn();


var router = express.Router();

// function fetch_models(req, res, next) {
//   db.all('SELECT * FROM models', function(err, rows) {
//     if (err) { return next(err); }
    
//     var models = rows.map(function(row) {
//       return {
//         id: row.id,
//         owner_id: row.owner_id,
//         title: row.title,
//         date: row.date,
//         stars: row.stars,
//         thumb_image_path: row.thumb_image_path,
//         path: row.path
//       }
//     });
//     res.locals.models = models;
//     next();
//   });
// }

function delete_model_from_db(model_id) {
    db.run('DELETE FROM models WHERE id = ?', model_id,
    function(err) {
        if (err) {
            return console.error(err.message);
        }
    });
  }

/* GET home page. */
// router.get('/delete', fetch_models, function(req, res, next) {
//   // console.log("index.cjs user: %s", req.user);
//   res.render('delete', { user: req.user });
// });

function fetch_models(req, res, next) {
    console.log("req.query.page: ", req.query.page);
    const page = req.query.page ? parseInt(req.query.page, 10) : 1;
    const itemsPerPage = 200; // 8;
    const offset = 0; // (page - 1) * itemsPerPage;
    // console.log("page: ", page);
    // console.log("offset: ", offset);

    let sql = 'SELECT COUNT(*) AS count FROM models';

    db.get(sql, [], (err, row) => {
        if (err) {
            throw err;
        }

        db.all('SELECT * FROM models ORDER BY id LIMIT ? OFFSET ?', [itemsPerPage, offset], (err, rows) => {
            if (err) {
                res.status(500).send(err.message);
                return;
            }
            // console.log(rows);
            // res.json(rows);
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
              res.locals.row_count = row.count;
              console.log('row count: ', res.locals.row_count);
              next();
        });
    });
}

router.get('/delete', fetch_models, (req, res) => {
  res.render('delete', { user: req.user });
// res.json(res.locals.models);
});


// Define a route for file upload.
router.post('/delete', (req, res) => {
    const row_data = req.body.row_data;
    console.log("row_data: ", row_data);
    const model_id = row_data[0]; // row_data.model;
    console.log("model_id: ", model_id);
    delete_model_from_db(model_id);
  
    // return res.render('upload', { user: req.user, upload: true });
    // return res.redirect('/upload');
    res.json({ message: 'Model is deleted!' });
  });

module.exports = router;
