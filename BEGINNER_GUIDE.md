# The "Explain Like I'm 5" Guide to Your Project

## 1. What is this project actually doing?
Imagine you are dropped into a huge, foggy mountain range at night. You have a GPS that tells you your **altitude** (height), but you can't see the landscape.
**Your Goal:** Find the absolute lowest point (the bottom of the deepest valley) in the entire mountain range.

*   **The Mountain Range** = The **CEC2017 Functions**. These are math equations that create "hills" and "valleys".
*   **The Altitude** = The **Error** or **Fitness**. We want this number to be as close to 0 as possible.
*   **You (The Hiker)** = The **Algorithm**.

## 2. The "Hikers" (Algorithms)
You have different teams of hikers trying to find the bottom. Each team has a different strategy:

*   **Genetic Algorithm (GA):**
    *   *Strategy:* "Evolution." You send out 100 hikers. You kill the ones standing on high peaks. You breed the ones standing in valleys to make "children" hikers who might be in even deeper valleys.
*   **Particle Swarm Optimization (PSO):**
    *   *Strategy:* "Follow the Leader." You send out 100 hikers with radios. If one hiker finds a deep hole, everyone else runs slightly towards them while keeping their own momentum.
*   **Harmony Search (HS):**
    *   *Strategy:* "Jazz Improvisation." Instead of hikers, imagine musicians trying to play the perfect song. They remember good notes (memory) and sometimes try a random new note (improvisation) to see if it sounds better.

## 3. How to Run the Code (The "Remote Control")
You are using **Streamlit**. It turns your Python code into a website so you don't have to type commands in a black screen.

1.  **The Sidebar (Left):**
    *   **Algorithm:** Pick your team (e.g., Harmony Search).
    *   **Function:** Pick the mountain range (F1, F2, etc.). Some are simple bowls (easy), some are crazy jagged rocks (hard).
    *   **Dimension (D):** How complex the map is. D=10 is a 10-dimensional map (hard). D=30 is super hard.
2.  **The Main Screen:**
    *   Click **"Run"**.
    *   Watch the **Graph**.
        *   **X-axis (Bottom):** Time/Effort (Evaluations).
        *   **Y-axis (Left):** Altitude (Error).
    *   **What you want to see:** The line crashing down quickly and hitting 0 (or getting very close).

## 4. Cheat Sheet for Questions
*   **Q: What is "Convergence"?**
    *   A: It's the line on the graph going down. It means the algorithm is "learning" or finding better solutions.
*   **Q: Why CEC2017?**
    *   A: It's like the Olympics for these algorithms. It's a standard test so we can fairly compare our code against others.
*   **Q: What is "Global Optima"?**
    *   A: The absolute deepest point in the map. Sometimes algorithms get stuck in a small hole (Local Optima) and think they finished, but there is a deeper hole nearby.
*   **Q: Why is the line flat?**
    *   A: The algorithm got stuck! It thinks it found the best spot, or it's moving too slowly to find a better one.

## 5. Files in your folder
*   `app.py`: The website code.
*   `hs.py`: The Harmony Search logic.
*   `others.py`: The other algorithms (PSO, GA, etc.).
