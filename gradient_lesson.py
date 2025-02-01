import streamlit as st
import numpy as np
import plotly.express as px
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize session state for page navigation if it doesn't exist
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Welcome"

# Set page config
st.set_page_config(page_title="Gradient Learning Module", layout="wide")

# יצירת תפריט ניווט
st.sidebar.title("Gradient Learning Module")
page = st.sidebar.radio("Select a chapter:", [
    "Welcome", "Introduction", "Derivatives Basics", "Gradient Explanation", 
    "Gradient Visualization", "Gradient Applications"],
    index=[
        "Welcome", "Introduction", "Derivatives Basics", "Gradient Explanation", 
        "Gradient Visualization", "Gradient Applications"
    ].index(st.session_state["current_page"]))

# Update current_page based on radio selection
st.session_state["current_page"] = page

# Add this helper function at the top of your file, after imports
def add_navigation_buttons(prev_page=None, next_page=None):
    col1, col2 = st.columns([1,1])
    
    with col1:
        if prev_page:
            if st.button(f"← Back to {prev_page}"):
                st.session_state["current_page"] = prev_page
                st.rerun()
    
    with col2:
        if next_page:
            button_text = "Take Quiz" if next_page == "Quiz" else f"Next: {next_page} →"
            if st.button(button_text):
                st.session_state["current_page"] = next_page
                st.rerun()

# Helper function to get derivative string
def get_derivative_string(func_choice):
    derivatives = {
        "f(x) = x²": "f'(x) = 2x",
        "f(x) = x³": "f'(x) = 3x²",
        "f(x) = sin(x)": "f'(x) = cos(x)",
        "f(x) = e^x": "f'(x) = e^x",
        "f(x) = ln(x)": "f'(x) = 1/x"
    }
    return derivatives.get(func_choice, "")

if st.session_state["current_page"] == "Welcome":
    st.title("Welcome to the Gradient Learning Module! 👋")
    
    st.markdown("""
    ### Start Your Journey into Understanding Gradients
    
    This interactive module will help you understand the concept of gradients from the ground up.
    
    #### The Big Picture
    For a function f(x₁, x₂, ..., xₙ), the gradient is a vector of all partial derivatives:
    
    $$
    \\nabla f = \\left(\\frac{\\partial f}{\\partial x_1}, \\frac{\\partial f}{\\partial x_2}, ..., \\frac{\\partial f}{\\partial x_n}\\right)
    $$
    
    Don't worry if this looks complicated - we'll break it down step by step!
    
    #### What You'll Learn:
    1. 📚 Basic concepts and intuition behind gradients
    2. 📐 How to calculate derivatives and partial derivatives
    3. 🎯 Understanding gradient direction and magnitude
    4. 💻 Real-world applications in machine learning and optimization
    
    #### How to Use This Module:
    - Navigate through sections using the sidebar menu
    - Try the interactive examples in each section
    - Test your knowledge with the final quiz
    - Take your time to understand each concept before moving forward
    
    #### Prerequisites:
    - Basic understanding of functions
    - Familiarity with basic algebra
    - Curiosity to learn! 🚀
    
    Ready to begin? Click 'Next' to start with the Introduction!
    """)
    
    add_navigation_buttons(next_page="Introduction")

elif st.session_state["current_page"] == "Introduction":
    st.title("Introduction to Gradient")
    
    # Add mountain climbing GIF in a column to control its size
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("images/hiker_gradient.gif", use_container_width=True)

    st.markdown("""
    ## What is a Gradient?
    The **gradient** is a vector that represents the direction and rate of the steepest ascent of a scalar function.
    
    ### General Form
    For any function with n variables f(x₁, x₂, ..., xₙ), the gradient is defined as:
    
    $$
    \\nabla f = \\left(\\frac{\\partial f}{\\partial x_1}, \\frac{\\partial f}{\\partial x_2}, ..., \\frac{\\partial f}{\\partial x_n}\\right)
    $$
    
    ### Special Case: Two Variables
    For the common case of a function f(x, y) with two variables:
    
    $$
    \\nabla f = \\left(\\frac{\\partial f}{\\partial x}, \\frac{\\partial f}{\\partial y}\\right)
    $$
    """)

    st.markdown("""
    ### Intuitive Example
    Imagine you are hiking on this mountain:
    - Your **altitude** is given by a function **f(x, y)**.
    - The **gradient** tells you which direction is the steepest ascent.
    - If you want to descend as quickly as possible, go **opposite to the gradient**.
    
    ### Applications of Gradient
    - **Mathematics & Physics** – Understanding spatial changes of functions.
    - **Computer Graphics** – Adjusting shading and lighting in 3D models.
    - **Machine Learning** – Used in **Gradient Descent** to optimize models.
    
    ### Summary
    ✅ The gradient is a powerful mathematical tool.\n
    ✅ It points in the direction of the greatest change.\n     
    ✅ It has applications in **machine learning, physics, and optimization**.
    
    ### Next Step
    To understand how gradients are computed, we need to review **partial derivatives**.
    """)
    
    # Add collapsible quiz section
    with st.expander("### Quick Check ✍️", expanded=False):
        st.markdown("Let's test your understanding of the basic gradient concepts:")
        
        # Quiz questions with empty default state
        q1 = st.radio(
            "1. What does the gradient represent?",
            ["The average rate of change", 
             "The direction of steepest descent",
             "The direction of steepest ascent",
             "The second derivative"],
            index=None,  # This makes no option selected by default
            key="q1"    # Unique key for the radio button
        )
        
        q2 = st.radio(
            "2. For a function f(x,y), how many components does its gradient have?",
            ["1", "2", "3", "4"],
            index=None,
            key="q2"
        )
        
        q3 = st.radio(
            "3. In machine learning, gradient descent moves:",
            ["In the direction of the gradient",
             "Opposite to the direction of the gradient",
             "Perpendicular to the gradient",
             "None of the above"],
            index=None,
            key="q3"
        )
        
        if st.button("Check Your Understanding", key="check_quiz"):
            if None in [q1, q2, q3]:
                st.warning("Please answer all questions before checking.")
            else:
                score = 0
                if q1 == "The direction of steepest ascent":
                    score += 1
                    st.success("Question 1: Correct! ✅")
                else:
                    st.error("Question 1: Incorrect ❌")
                    
                if q2 == "2":
                    score += 1
                    st.success("Question 2: Correct! ✅")
                else:
                    st.error("Question 2: Incorrect ❌")
                    
                if q3 == "Opposite to the direction of the gradient":
                    score += 1
                    st.success("Question 3: Correct! ✅")
                else:
                    st.error("Question 3: Incorrect ❌")
                    
                st.markdown(f"### Your Score: {score}/3")
                if score == 3:
                    st.balloons()
                    st.success("Perfect! You're ready to move on to the next section! 🎉")
                elif score >= 2:
                    st.success("Good understanding! Review the concepts you missed and continue! 👍")
                else:
                    st.info("Take some time to review the concepts above before moving forward. 📚")

    add_navigation_buttons(prev_page="Welcome", next_page="Derivatives Basics")

elif st.session_state["current_page"] == "Derivatives Basics":
    st.title("Derivatives - Pre-requisite for Gradient")
    st.markdown("""
    ## Understanding Derivatives
    A **derivative** measures the instantaneous rate of change of a function with respect to one of its variables.
    
    ### Definition
    For a function **f(x)**, its derivative represents the slope of the tangent line at any point:
    
    $$
    f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h} = \\lim_{\\Delta x \\to 0} \\frac{\\Delta y}{\\Delta x}
    $$
    
    ### Differentiation Rules
    1. **Power Rule:**
       $$\\frac{d}{dx}[x^n] = nx^{n-1}$$
    
    2. **Sum Rule:**
       $$\\frac{d}{dx}[f(x) ± g(x)] = f'(x) ± g'(x)$$
    
    3. **Product Rule:**
       $$\\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$$
    
    4. **Quotient Rule:**
       $$\\frac{d}{dx}[\\frac{f(x)}{g(x)}] = \\frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$$
    
    5. **Chain Rule:**
       $$\\frac{d}{dx}[f(g(x))] = f'(g(x))g'(x)$$
    
    ### Common Derivatives
    $$
    \\begin{array}{ll}
    \\frac{d}{dx}[c] = 0 & \\text{(constant rule)} \\\\[1em]
    \\frac{d}{dx}[x] = 1 & \\text{(power rule, n=1)} \\\\[1em]
    \\frac{d}{dx}[\\sin(x)] = \\cos(x) & \\frac{d}{dx}[\\cos(x)] = -\\sin(x) \\\\[1em]
    \\frac{d}{dx}[\\tan(x)] = \\sec^2(x) & \\frac{d}{dx}[\\sec(x)] = \\sec(x)\\tan(x) \\\\[1em]
    \\frac{d}{dx}[e^x] = e^x & \\frac{d}{dx}[\\ln(x)] = \\frac{1}{x} \\\\[1em]
    \\frac{d}{dx}[a^x] = a^x\\ln(a) & \\frac{d}{dx}[\\log_a(x)] = \\frac{1}{x\\ln(a)}
    \\end{array}
    $$
    """)

    # Examples section with expandable content
    with st.expander("📝 Example Derivatives", expanded=False):
        st.markdown("""
        ### Step-by-Step Examples
        
        1. **Product Rule Example:** f(x) = x²sin(x)
           ```
           f'(x) = x² · d/dx[sin(x)] + sin(x) · d/dx[x²]
                 = x²cos(x) + sin(x)(2x)
                 = x²cos(x) + 2xsin(x)
           ```
        
        2. **Chain Rule Example:** f(x) = sin(x²)
           ```
           f'(x) = cos(x²) · d/dx[x²]
                 = cos(x²)(2x)
                 = 2xcos(x²)
           ```
        
        3. **Quotient Rule Example:** f(x) = tan(x) = sin(x)/cos(x)
           ```
           f'(x) = [cos(x)cos(x) - sin(x)(-sin(x))]/cos²(x)
                 = [cos²(x) + sin²(x)]/cos²(x)
                 = sec²(x)
           ```
        """)

    # Practice section
    with st.expander("✍️ Practice Problems", expanded=False):
        st.markdown("### Try these derivatives:")
        
        practice_problems = {
            "1": {"expr": "x⁴ + 3x² - 2x", "solution": "4x³ + 6x - 2"},
            "2": {"expr": "sin(x)cos(x)", "solution": "cos²(x) - sin²(x)"},
            "3": {"expr": "e^(x²)", "solution": "2xe^(x²)"},
            "4": {"expr": "(x² + 1)/(x - 1)", "solution": "(2x(x-1) - (x²+1))/(x-1)²"},
            "5": {"expr": "ln(x²+1)", "solution": "2x/(x²+1)"}
        }
        
        for num, prob in practice_problems.items():
            col1, col2 = st.columns([3,1])
            with col1:
                st.latex(f"\\frac{d}{{dx}}[{prob['expr']}]")
            with col2:
                if st.button(f"Show Solution {num}"):
                    st.latex(prob['solution'])

    # Interactive visualization section follows...

elif st.session_state["current_page"] == "Gradient Explanation":
    st.title("Detailed Explanation of Gradient")
    st.markdown("""
    ### Understanding the Gradient
    The gradient is a vector of all partial derivatives of a function. For a function f(x,y):
    
    $$
    \\nabla f = \\left(\\frac{\\partial f}{\\partial x}, \\frac{\\partial f}{\\partial y}\\right)
    $$
    
    ### Example
    For the function f(x,y) = x² + y², the gradient is:
    
    $$
    \\nabla f = (2x, 2y)
    $$
    
    ### Key Properties
    1. 📈 Points in direction of steepest increase
    2. 📏 Magnitude shows rate of increase
    3. ⚡ Perpendicular to level curves
    
    ### Interactive Calculator
    Try calculating a gradient yourself:
    """)
    
    # Interactive gradient calculator
    x, y = sp.symbols('x y')
    input_function = st.text_input("Enter a function in x and y (e.g., x**2 + y**2):", "x**2 + y**2")
    
    try:
        expr = sp.sympify(input_function)
        grad_x = sp.diff(expr, x)
        grad_y = sp.diff(expr, y)
        st.markdown(f"""
        The gradient of f(x,y) = {expr} is:
        
        $$
        \\nabla f = ({grad_x}, {grad_y})
        $$
        """)
    except:
        st.error("Please enter a valid mathematical expression")
    
    add_navigation_buttons(prev_page="Derivatives Basics", next_page="Gradient Visualization")

elif st.session_state["current_page"] == "Gradient Visualization":
    st.title("Visualizing Gradient")
    st.markdown("""
    Below is a graphical representation of gradient fields.
    """)
    
    add_navigation_buttons(prev_page="Gradient Explanation", next_page="Gradient Applications")

elif st.session_state["current_page"] == "Gradient Applications":
    st.title("Applications of Gradient in Computer Science")
    st.markdown("""
    ### Applications
    - **Gradient Descent**: Used in machine learning for optimization.
    - **Physics**: Used in potential field calculations.
    - **Mathematical Modeling**: Used in function optimization problems.
    """)
    
    add_navigation_buttons(prev_page="Gradient Visualization", next_page="Quiz")
