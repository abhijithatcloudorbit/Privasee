import express from "express";
import cors from "cors";
import session from "express-session";
import dotenv from "dotenv";
import passport from "passport";

import configurePassport from "./passport-config.js";
import authRoutes from "./routes/auth.js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

// --- middlewares ---
app.use(
  cors({
    origin: process.env.FRONTEND_URL,
    credentials: true,
  })
);

app.use(express.json());

app.use(
  session({
    secret: process.env.SESSION_SECRET,
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false },
  })
);

// passport
configurePassport();
app.use(passport.initialize());
app.use(passport.session());

// routes
app.use("/auth", authRoutes);

app.get("/", (req, res) =>
  res.send("Privasee Auth Backend Running Successfully")
);

app.listen(PORT, () =>
  console.log(`Backend running → ${process.env.BASE_URL}`)
);
