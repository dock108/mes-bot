const express = require('express');
const { transformText } = require('../controllers/transformController');

const router = express.Router();

router.post('/transform', transformText);

module.exports = router; 