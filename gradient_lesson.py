import streamlit as st
import numpy as np
import plotly.express as px
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from derivative_examples import DERIVATIVE_RULES, PRACTICE_PROBLEMS

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
    A **derivative** measures the rate at which a function changes with respect to one of its variables.
    
    ### Definition and Interpretation
    If we have a function **f(x)**, its derivative is defined as:
    
    $$
    f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}
    $$
    
    This represents:
    1. The instantaneous rate of change at any point
    2. The slope of the tangent line at point x
    3. The best linear approximation of the function near x
    
    ### Fundamental Rules of Differentiation
    """)
    
    # Create tabs for different rule categories
    rules_tab, examples_tab, practice_tab = st.tabs(["Rules", "Examples", "Practice"])
    
    with rules_tab:
        st.markdown("""
        ### Basic Rules & Common Functions
        $$
        \\begin{align*}
        & \\textbf{Power:} & \\frac{d}{dx} x^n &= nx^{n-1} & & \\textbf{Exp/Log:} & \\frac{d}{dx}e^x &= e^x & \\frac{d}{dx}\\ln x &= \\frac{1}{x} \\\\[0.7em]
        & \\textbf{Product:} & \\frac{d}{dx}[fg] &= f'g + fg' & & \\textbf{Trig:} & \\frac{d}{dx}\\sin x &= \\cos x & \\frac{d}{dx}\\cos x &= -\\sin x \\\\[0.7em]
        & \\textbf{Quotient:} & \\frac{d}{dx}\\frac{f}{g} &= \\frac{f'g - fg'}{g^2} \\\\[0.7em]
        & \\textbf{Chain:} & \\frac{d}{dx}f(g(x)) &= f'(g(x))g'(x)
        \\end{align*}
        $$
        """)
    
    with examples_tab:
        selected_rule = st.selectbox(
            "Select a rule to see examples:",
            list(DERIVATIVE_RULES.keys())
        )
        
        st.markdown("### Examples:")
        for example in DERIVATIVE_RULES[selected_rule]["examples"]:
            st.markdown(rf"""
            $$
            \frac{{d}}{{dx}}({example['input']}) = {example['output']}
            $$
            """)
    
    with practice_tab:
        st.markdown("### Practice Problems")
        st.markdown("Click any problem tile to see its solution. Click again to close.")
        
        # Initialize problem page in session state if not exists
        if 'problem_page' not in st.session_state:
            st.session_state.problem_page = 0
            
        # Calculate total number of pages
        total_pages = (len(PRACTICE_PROBLEMS) + 5) // 6  # Ceiling division
        
        # Display current set of 6 problems (2 rows of 3)
        start_idx = st.session_state.problem_page * 6
        for row in range(2):  # 2 rows
            cols = st.columns(3)  # 3 columns per row
            for col in range(3):  # Fill each column
                prob_idx = start_idx + row * 3 + col
                if prob_idx < len(PRACTICE_PROBLEMS):
                    problem = PRACTICE_PROBLEMS[prob_idx]
                    with cols[col]:
                        with st.expander(f"Problem {prob_idx + 1}:\nf(x) = ${problem['function']}$", expanded=False):
                            st.latex(rf"\frac{{d}}{{dx}}({problem['function']}) = {problem['solution']}")
                            st.markdown(f"**Explanation:**\n{problem['explanation']}")
        
        # Navigation buttons with compact layout
        col1, col2, col3, col4 = st.columns([8, 1, 1, 1])
        
        # Empty column for spacing
        with col1:
            st.write("")
            
        # Previous button
        with col2:
            if st.session_state.problem_page > 0:
                if st.button("← Prev", use_container_width=True):
                    st.session_state.problem_page -= 1
                    st.rerun()
                    
        # Page number
        with col3:
            st.markdown(f"<div style='text-align: center; margin-top: 5px;'>{st.session_state.problem_page + 1}/{total_pages}</div>", unsafe_allow_html=True)
            
        # Next button
        with col4:
            if st.session_state.problem_page < total_pages - 1:
                if st.button("Next →", use_container_width=True):
                    st.session_state.problem_page += 1
                    st.rerun()

    st.markdown("### Interactive Visualization")
    # Add function selector in a smaller column
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        function_choice = st.selectbox(
            "Select a function:",  # Shortened label
            ["f(x) = x²", 
             "f(x) = x³", 
             "f(x) = sin(x)",
             "f(x) = e^x",
             "f(x) = ln(x)"],
            index=0
        )

    # Display the selected function and its derivative
    st.markdown(f"""
    - Derivative: {get_derivative_string(function_choice)}
    """)

    # Create interactive plot with slider
    x_point = st.slider("Select x position", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    
    # Calculate function values based on selection
    X = np.linspace(-2, 2, 100)
    
    if function_choice == "f(x) = x²":
        Y = X**2
        dY = 2*X
        y_point = x_point**2
        derivative_at_point = 2 * x_point
    elif function_choice == "f(x) = x³":
        Y = X**3
        dY = 3*(X**2)
        y_point = x_point**3
        derivative_at_point = 3 * (x_point**2)
    elif function_choice == "f(x) = sin(x)":
        Y = np.sin(X)
        dY = np.cos(X)
        y_point = np.sin(x_point)
        derivative_at_point = np.cos(x_point)
    elif function_choice == "f(x) = e^x":
        Y = np.exp(X)
        dY = np.exp(X)
        y_point = np.exp(x_point)
        derivative_at_point = np.exp(x_point)
    else:  # ln(x)
        # Adjust domain for ln(x) since it's only defined for x > 0
        X = np.linspace(0.1, 2, 100)
        Y = np.log(X)
        dY = 1/X
        y_point = np.log(max(x_point, 0.1))
        derivative_at_point = 1/max(x_point, 0.1)
        x_point = max(x_point, 0.1)  # Ensure x is positive for ln(x)

    # Create points for tangent line
    x_tangent = np.array([x_point - 0.5, x_point + 0.5])
    y_tangent = derivative_at_point * (x_tangent - x_point) + y_point
    
    # Create subplots
    fig = go.Figure()
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=(
                           f"Function {function_choice} with tangent line (slope: {derivative_at_point:.2f})",
                           f"Derivative d/dx({function_choice})"
                       ))
    
    # Add original function to first subplot
    fig.add_trace(
        go.Scatter(x=X, y=Y, mode='lines', name=function_choice, line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add tangent line to first subplot (now solid)
    fig.add_trace(
        go.Scatter(x=x_tangent, y=y_tangent, mode='lines', 
                  name=f'Tangent at x={x_point}', 
                  line=dict(color='red'),
                  showlegend=False),  # Removed dash='dash'
        row=1, col=1
    )
    
    # Add point on original function
    fig.add_trace(
        go.Scatter(x=[x_point], y=[y_point], mode='markers',
                  name=f'x = {x_point}', 
                  marker=dict(color='red', size=10)),
        row=1, col=1
    )
    
    # Add derivative function to second subplot
    fig.add_trace(
        go.Scatter(x=X, y=dY, mode='lines', 
                  name=f"d/dx({function_choice}) = {derivative_at_point:.2f}",
                  line=dict(color='orange'),
                  showlegend=False),
        row=2, col=1
    )
    
    # Add horizontal dotted line at derivative value
    fig.add_trace(
        go.Scatter(x=[-2, 2], y=[derivative_at_point, derivative_at_point],
                  mode='lines',
                  name=f"d/dx({function_choice}) = {derivative_at_point:.2f}",
                  line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    
    # Add point on derivative
    fig.add_trace(
        go.Scatter(x=[x_point], y=[derivative_at_point], 
                  mode='markers',
                  name='Derivative Value', 
                  marker=dict(color='red', size=10),
                  showlegend=False),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text="Function and its Derivative"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_yaxes(title_text="f'(x)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    add_navigation_buttons(prev_page="Introduction", next_page="Gradient Explanation")

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
