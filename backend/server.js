// server.js
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const session = require("express-session");
const passport = require("passport");
const cookieParser = require("cookie-parser");
const configurePassport = require("./passport-config");
const authRoutes = require("./routes/auth");

const app = express();
const PORT = process.env.PORT || 4000;

app.use(cors({
  origin: process.env.FRONTEND_URL,
  credentials: true,
}));
app.use(express.json());
app.use(cookieParser());

// minimal session (not strictly required since we use JWT)
app.use(session({
  secret: process.env.SESSION_SECRET || "dev_secret_change_me",
  resave: false,
  saveUninitialized: true,
  cookie: { secure: false }, // set secure: true if using HTTPS
}));

// passport
configurePassport();
app.use(passport.initialize());
app.use(passport.session());

// routes
app.use("/auth", authRoutes);

app.get("/", (req, res) => res.send("Privasee Auth Backend running"));

app.listen(PORT, () => {
  console.log(`Auth server listening on ${process.env.BASE_URL || `http://localhost:${PORT}`}`);
});
