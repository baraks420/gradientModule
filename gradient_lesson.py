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
    "Gradient Visualization", "Gradient Applications", "Quiz"],
    index=[
        "Welcome", "Introduction", "Derivatives Basics", "Gradient Explanation", 
        "Gradient Visualization", "Gradient Applications", "Quiz"
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
    st.markdown("""
    - Derivative: """ + get_derivative_string(function_choice) + """
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
    
    # Create tabs for different sections
    theory_tab, examples_tab, practice_tab, calculator_tab = st.tabs(["Theory", "Examples", "Practice", "Calculator"])
    
    with theory_tab:
    st.markdown("""
    ## Mathematical Foundation of Gradients

    ### Definition
    For a multivariable function f: ℝⁿ → ℝ, the gradient is defined as the vector of all partial derivatives:

    $$
    \\nabla f = \\begin{pmatrix} 
    \\frac{\\partial f}{\\partial x_1} \\\\
    \\frac{\\partial f}{\\partial x_2} \\\\
    \\vdots \\\\
    \\frac{\\partial f}{\\partial x_n}
    \\end{pmatrix}
    $$

    ### Key Theoretical Properties
    - **Directional Derivative**: 
      $$D_v f(x) = \\nabla f(x) \\cdot v$$
    - **Gradient = direction of steepest ascent**
    - **Magnitude** of the gradient = how steep the ascent is
    - **Orthogonal to level sets**

    ---
    ### 🔍 Example: \( f(x, y) = x^2 + y^2 \)

    #### 🧮 Step 1: Compute the Gradient

    $$
    \\frac{\\partial f}{\\partial x} = 2x \\quad , \\quad \\frac{\\partial f}{\\partial y} = 2y
    $$

    So the gradient is:

    $$
    \\nabla f(x, y) = (2x,\\ 2y)
    $$

    ---

    #### 🧷 Step 2: Plug in Some Points

    **Example A**:
    $$
    (x, y) = (1, 2) \\Rightarrow \\nabla f = (2, 4)
    $$

    **Example B**:
    $$
    (x, y) = (-3, 1) \\Rightarrow \\nabla f = (-6, 2)
    $$

    ---

    #### 🔁 Step 3: What Do These Numbers Mean?

    - The gradient is a **vector** showing the direction of **steepest increase** of the function.
    - For Example A:
        - The vector (2, 4) points "up and to the right".
        - This means the function increases fastest in that direction.
        - Its magnitude:
          $$
          \\|\\nabla f\\| = \\sqrt{2^2 + 4^2} = \\sqrt{20} \\approx 4.47
          $$

    - For Example B:
        - The vector (-6, 2) points "left and slightly up".
        - It means the function increases fastest in that direction, and even faster than in Example A (because magnitude is larger).

    ---

    #### 🧭 Summary:
    - The **direction** of the gradient shows how to climb uphill fastest.
    - The **length** of the gradient tells how steep the climb is.
    - The farther you are from the origin in this function, the steeper the slope becomes.

    """)

    with examples_tab:
        st.markdown("""
        ## Gradient Examples
        
        ### Basic Rules for Partial Derivatives
        
        1. **Treat other variables as constants**:
           When finding ∂f/∂x, treat y as a constant, and vice versa.
        
        2. **Power Rule**:
           $$\\frac{\\partial}{{\\partial x}} x^n = nx^{n-1}$$
        
        3. **Product Rule**:
           $$\\frac{\\partial}{{\\partial x}} (uv) = u\\frac{\\partial v}{\\partial x} + v\\frac{\\partial u}{\\partial x}$$
        
        ### Common Examples
        """)
        
        # Example selector
        example_function = st.selectbox(
            "Select an example to see its gradient:",
            [
                "f(x,y) = x² + y²",
                "f(x,y) = x²y",
                "f(x,y) = sin(x)cos(y)",
                "f(x,y) = e^(x+y)",
                "f(x,y) = ln(x) + y²"
            ]
        )
        
        examples = {
            "f(x,y) = x² + y²": {
                "gradient": ["2x", "2y"],
                "explanation": "Each partial derivative treats the other variable as a constant. For ∂f/∂x, y² is constant; for ∂f/∂y, x² is constant."
            },
            "f(x,y) = x²y": {
                "gradient": ["2xy", "x²"],
                "explanation": "Use the product rule for ∂f/∂x. For ∂f/∂y, treat x² as a constant coefficient."
            },
            "f(x,y) = sin(x)cos(y)": {
                "gradient": ["cos(x)cos(y)", "-sin(x)sin(y)"],
                "explanation": "Use the product rule and chain rule. Note the negative sign in ∂f/∂y due to the derivative of cos(y)."
            },
            "f(x,y) = e^(x+y)": {
                "gradient": ["e^(x+y)", "e^(x+y)"],
                "explanation": "The chain rule gives us the same result for both partial derivatives since e^(x+y) is symmetric in x and y."
            },
            "f(x,y) = ln(x) + y²": {
                "gradient": ["1/x", "2y"],
                "explanation": "The partial derivatives are independent since the function is a sum. Use the natural log rule for x and power rule for y."
            }
        }
        
        example = examples[example_function]
        st.markdown(f"""
        For {example_function}, the gradient is:
        
        $$
        \\nabla f = \\begin{{pmatrix}} 
        \\frac{{\\partial f}}{{\\partial x}} = {example['gradient'][0]} \\\\[1em]
        \\frac{{\\partial f}}{{\\partial y}} = {example['gradient'][1]}
        \\end{{pmatrix}}
        $$
        
        **Explanation**: {example['explanation']}
        """)

    with practice_tab:
        st.markdown("""
        ## Practice Problems
        
        Test your understanding of gradients with these practice problems.
        Click each problem to see its solution.
        """)
        
        # Initialize practice problem state if not exists
        if 'gradient_problem_page' not in st.session_state:
            st.session_state.gradient_problem_page = 0
        
        practice_problems = [
            {
                "function": "f(x,y) = x³ + 2xy",
                "gradient": ["3x² + 2y", "2x"],
                "explanation": "For ∂f/∂x, use power rule on x³ and treat y as constant in 2xy. For ∂f/∂y, treat x as constant."
            },
            {
                "function": "f(x,y) = xy² + sin(x)",
                "gradient": ["y² + cos(x)", "2xy"],
                "explanation": "For ∂f/∂x, y² is a constant coefficient. For ∂f/∂y, use power rule treating x as constant."
            },
            {
                "function": "f(x,y) = e^x cos(y)",
                "gradient": ["e^x cos(y)", "-e^x sin(y)"],
                "explanation": "Use product rule. Note the negative sign in ∂f/∂y from the derivative of cos(y)."
            },
            {
                "function": "f(x,y) = ln(x²+y²)",
                "gradient": ["2x/(x²+y²)", "2y/(x²+y²)"],
                "explanation": "Use chain rule. The derivative of ln(u) is 1/u times the derivative of u."
            },
            {
                "function": "f(x,y) = x²y³",
                "gradient": ["2xy³", "3x²y²"],
                "explanation": "Use product rule and power rule. Treat other variables as constants when taking each partial derivative."
            }
        ]
        
        # Display current problem
        current_problem = practice_problems[st.session_state.gradient_problem_page]
        
        with st.expander(f"Problem {st.session_state.gradient_problem_page + 1}: Find ∇f for {current_problem['function']}", expanded=True):
            if st.button("Show Solution", key=f"sol_{st.session_state.gradient_problem_page}"):
                st.markdown(f"""
                The gradient is:
                $$
                \\nabla f = \\begin{{pmatrix}} 
                {current_problem['gradient'][0]} \\\\[1em]
                {current_problem['gradient'][1]}
                \\end{{pmatrix}}
                $$
                
                **Explanation**: {current_problem['explanation']}
                """)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.session_state.gradient_problem_page > 0:
                if st.button("← Previous Problem"):
                    st.session_state.gradient_problem_page -= 1
                    st.rerun()
        
        with col2:
            st.markdown(f"<div style='text-align: center'>Problem {st.session_state.gradient_problem_page + 1} of {len(practice_problems)}</div>", unsafe_allow_html=True)
        
        with col3:
            if st.session_state.gradient_problem_page < len(practice_problems) - 1:
                if st.button("Next Problem →"):
                    st.session_state.gradient_problem_page += 1
                    st.rerun()

    with calculator_tab:
        st.markdown("""
        ### Interactive Gradient Calculator
        Experiment with calculating gradients of different functions:
        """)
        
        # Interactive gradient calculator with advanced options
        x, y, z = sp.symbols('x y z')
        
        # Function input
        input_function = st.text_input("Enter a function (e.g., x**2 + y**2, x*y*z):", "x**2 + y**2")
        
        # Variable selection
        variables = st.multiselect(
            "Select variables to differentiate with respect to:",
            ['x', 'y', 'z'],
            default=['x', 'y']
        )
        
        try:
            expr = sp.sympify(input_function)
            gradient_components = []
            for var in variables:
                grad = sp.diff(expr, eval(var))
                gradient_components.append(grad)
            
            # Display the gradient
            gradient_expr = ' \\\\ '.join([str(comp) for comp in gradient_components])
            st.markdown("""
            For the function f(""" + ', '.join(variables) + ") = " + str(expr) + """:
            
            The gradient is:
            $$
            \\nabla f = \\begin{pmatrix} """ + gradient_expr + """ \\end{pmatrix}
            $$
            """)
            
            # Additional mathematical properties
            if len(variables) == 2:  # Only for 2D functions
                x_val = st.slider("x value", -5.0, 5.0, 0.0, 0.1)
                y_val = st.slider("y value", -5.0, 5.0, 0.0, 0.1)
                
                # Calculate gradient magnitude at point
                grad_magnitude = sp.sqrt(sum(comp**2 for comp in gradient_components))
                magnitude_at_point = grad_magnitude.subs({x: x_val, y: y_val})
                
                st.markdown("""
                At point (""" + str(x_val) + ", " + str(y_val) + """):
                
                Gradient magnitude: """ + f"{magnitude_at_point:.2f}" + """
                """)
                
        except Exception as e:
            st.error(f"Please enter a valid mathematical expression. Error: {str(e)}")
    
    add_navigation_buttons(prev_page="Derivatives Basics", next_page="Gradient Visualization")

elif st.session_state["current_page"] == "Gradient Visualization":
    st.title("Visualizing Gradients in Multiple Dimensions")
    st.markdown("""
    ### Understanding Gradient Fields
    A gradient field is a visual representation of gradients at different points in space. 
    The arrows in the field indicate:
    - **Direction**: Where the function increases most rapidly
    - **Length**: How steep the increase is at that point
    
    ### Interactive Gradient Field Visualization
    Select a function to visualize its gradient field:
    """)
    
    # Function selection
    function_choice = st.selectbox(
        "Choose a function:",
        ["f(x,y) = x² + y²", 
         "f(x,y) = sin(x)cos(y)",
         "f(x,y) = x² - y²",
         "f(x,y) = e^(-x² - y²)"],
        index=0
    )
    
    # Create grid of points
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function values and gradients based on selection
    if function_choice == "f(x,y) = x² + y²":
        Z = X**2 + Y**2
        U = 2*X  # dx
        V = 2*Y  # dy
        title = "Gradient Field of f(x,y) = x² + y²"
    elif function_choice == "f(x,y) = sin(x)cos(y)":
        Z = np.sin(X) * np.cos(Y)
        U = np.cos(X) * np.cos(Y)  # dx
        V = -np.sin(X) * np.sin(Y)  # dy
        title = "Gradient Field of f(x,y) = sin(x)cos(y)"
    elif function_choice == "f(x,y) = x² - y²":
        Z = X**2 - Y**2
        U = 2*X  # dx
        V = -2*Y  # dy
        title = "Gradient Field of f(x,y) = x² - y²"
    else:  # e^(-x² - y²)
        Z = np.exp(-X**2 - Y**2)
        U = -2*X*np.exp(-X**2 - Y**2)  # dx
        V = -2*Y*np.exp(-X**2 - Y**2)  # dy
        title = "Gradient Field of f(x,y) = e^(-x² - y²)"
    
    # Create tabs for different visualizations
    contour_tab, field_tab, surface_tab = st.tabs(["Contour Plot", "Gradient Field", "3D Surface"])
    
    with contour_tab:
        st.markdown("""
        ### Contour Plot with Gradient Vectors
        The contour lines show points of equal height (level curves). 
        The gradient vectors are always perpendicular to these contour lines.
        """)
        
        fig = go.Figure()
        
        # Add contour plot
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z,
            colorscale='Viridis',
            showscale=True,
            name='Function Value'
        ))
        
        # Add gradient vectors using quiver plot
        skip = 2  # Show fewer arrows for clarity
        # Normalize vectors for better visualization
        magnitudes = np.sqrt(U**2 + V**2)
        max_magnitude = np.max(magnitudes)
        scale = 0.2  # Scale factor for arrow length
        
        for i in range(0, len(x), skip):
            for j in range(0, len(y), skip):
                # Starting point of arrow
                x_start = X[i,j]
                y_start = Y[i,j]
                
                # Calculate arrow end point
                magnitude = magnitudes[i,j]
                if magnitude > 0:  # Avoid division by zero
                    dx = U[i,j] / magnitude * scale
                    dy = V[i,j] / magnitude * scale
                else:
                    dx = dy = 0
                
                # Add arrow
                fig.add_trace(go.Scatter(
                    x=[x_start, x_start + dx],
                    y=[y_start, y_start + dy],
                    mode='lines',
                    line=dict(color='red', width=1),
                    showlegend=False
                ))
                
                # Add arrowhead
                fig.add_trace(go.Scatter(
                    x=[x_start + dx],
                    y=[y_start + dy],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        angle=np.arctan2(dy, dx) * 180 / np.pi - 90,
                        size=8,
                        color='red'
                    ),
                    showlegend=False
                ))
        
        # Add a single legend entry for gradient vectors
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines+markers',
            name='Gradient Vectors',
            line=dict(color='red'),
            marker=dict(symbol='triangle-up', color='red')
        ))
        
        fig.update_layout(
            title=title,
            width=700,
            height=600,
            xaxis=dict(range=[-2.2, 2.2]),
            yaxis=dict(range=[-2.2, 2.2]),
            xaxis_title='x',
            yaxis_title='y'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with field_tab:
        st.markdown("""
        ### Pure Gradient Field
        This visualization shows just the gradient vectors, helping you understand
        the direction and magnitude of steepest increase at each point.
        The color and length of each arrow represents the gradient magnitude.
        """)
        
        fig = go.Figure()
        
        # Create gradient field using arrows
        skip = 2
        # Normalize vectors for better visualization
        magnitudes = np.sqrt(U**2 + V**2)
        max_magnitude = np.max(magnitudes)
        scale = 0.2  # Scale factor for arrow length
        
        # Create a colormap function
        def get_color(magnitude):
            # Convert magnitude to a color using viridis colormap
            # Returns color in rgb format
            norm_magnitude = magnitude / max_magnitude
            return f'rgb({int(255 * (1-norm_magnitude))}, {int(255 * norm_magnitude)}, 255)'
        
        for i in range(0, len(x), skip):
            for j in range(0, len(y), skip):
                # Starting point of arrow
                x_start = X[i,j]
                y_start = Y[i,j]
                
                # Calculate arrow end point
                magnitude = magnitudes[i,j]
                if magnitude > 0:  # Avoid division by zero
                    dx = U[i,j] / magnitude * scale * magnitude/max_magnitude
                    dy = V[i,j] / magnitude * scale * magnitude/max_magnitude
                else:
                    dx = dy = 0
                
                # Get color based on magnitude
                arrow_color = get_color(magnitude)
                
                # Add arrow shaft
                fig.add_trace(go.Scatter(
                    x=[x_start, x_start + dx],
                    y=[y_start, y_start + dy],
                    mode='lines',
                    line=dict(
                        color=arrow_color,
                        width=2
                    ),
                    showlegend=False
                ))
                
                # Add arrowhead
                fig.add_trace(go.Scatter(
                    x=[x_start + dx],
                    y=[y_start + dy],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        angle=np.arctan2(dy, dx) * 180 / np.pi - 90,
                        size=8,
                        color=arrow_color
                    ),
                    showlegend=False
                ))
        
        # Add colorbar
        colorbar_trace = go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=1,
                color=[0, max_magnitude],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text='Gradient Magnitude',
                        side='right'
                    )
                )
            ),
            showlegend=False
        )
        fig.add_trace(colorbar_trace)
        
        fig.update_layout(
            title=title,
            width=700,
            height=600,
            xaxis=dict(range=[-2.2, 2.2]),
            yaxis=dict(range=[-2.2, 2.2]),
            xaxis_title='x',
            yaxis_title='y'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with surface_tab:
        st.markdown("""
        ### 3D Surface Plot
        This visualization shows the actual shape of the function in 3D space.
        The gradient vectors (shown as red arrows) at each point are tangent to the surface 
        and point in the direction of steepest ascent.
        """)
        
        fig = go.Figure()
        
        # Add surface plot
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.8))
        
        # Add gradient vectors as arrows
        skip = 4  # Show fewer arrows for clarity in 3D
        # Normalize vectors for better visualization
        magnitudes = np.sqrt(U**2 + V**2)
        max_magnitude = np.max(magnitudes)
        scale = 0.2  # Scale factor for arrow length
        
        # Create points for gradient vectors
        x_points = []
        y_points = []
        z_points = []
        u_points = []
        v_points = []
        w_points = []
        
        for i in range(0, len(x), skip):
            for j in range(0, len(y), skip):
                x_start = X[i,j]
                y_start = Y[i,j]
                z_start = Z[i,j]
                
                # Calculate gradient components
                dx = U[i,j]
                dy = V[i,j]
                # Calculate z-component based on the surface slope
                if i > 0 and i < len(x)-1 and j > 0 and j < len(y)-1:
                    dz = (Z[i+1,j] - Z[i-1,j])/(2*scale) * dx + (Z[i,j+1] - Z[i,j-1])/(2*scale) * dy
                else:
                    dz = 0
                
                # Normalize and scale the vector
                magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
                if magnitude > 0:
                    dx = dx/magnitude * scale
                    dy = dy/magnitude * scale
                    dz = dz/magnitude * scale
                
                x_points.append(x_start)
                y_points.append(y_start)
                z_points.append(z_start)
                u_points.append(dx)
                v_points.append(dy)
                w_points.append(dz)
        
        # Add arrows (lines + cones)
        for i in range(len(x_points)):
            # Base of the arrow
            x_base = x_points[i]
            y_base = y_points[i]
            z_base = z_points[i]
            
            # Vector components
            dx = u_points[i]
            dy = v_points[i]
            dz = w_points[i]
            
            # End point of the line (slightly before the cone)
            line_fraction = 0.8  # Line takes up 80% of the arrow length
            x_end = x_base + dx * line_fraction
            y_end = y_base + dy * line_fraction
            z_end = z_base + dz * line_fraction
            
            # Add line (arrow shaft)
            fig.add_trace(go.Scatter3d(
                x=[x_base, x_end],
                y=[y_base, y_end],
                z=[z_base, z_end],
                mode='lines',
                line=dict(color='red', width=3),
                showlegend=False
            ))
            
            # Add cone (arrowhead)
            fig.add_trace(go.Cone(
                x=[x_end],
                y=[y_end],
                z=[z_end],
                u=[dx * (1-line_fraction)],  # Cone size is the remaining 20%
                v=[dy * (1-line_fraction)],
                w=[dz * (1-line_fraction)],
                colorscale=[[0, 'red'], [1, 'red']],
                showscale=False,
                sizeref=0.25  # Smaller cones for better proportion
            ))
        
        fig.update_layout(
            title=title,
            width=700,
            height=600,
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='f(x,y)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='cube'
            ),
            showlegend=False
        )
        
        # Add camera controls explanation
        st.plotly_chart(fig, use_container_width=True)
        st.info("""
        💡 **Tip**: You can interact with the 3D plot:
        - Click and drag to rotate the view
        - Scroll to zoom in/out
        - Right-click and drag to pan
        - Double-click to reset the view
        """)
    
    st.markdown("""
    ### Key Observations
    1. **Gradient Direction**: 
       - The arrows always point in the direction of steepest increase
       - Longer arrows indicate steeper slopes
       
    2. **Level Curves**: 
       - The contour lines connect points of equal height
       - The gradient is always perpendicular to the level curves
       
    3. **Critical Points**:
       - Notice how the gradient vectors become shorter near critical points
       - At local maxima/minima, the gradient approaches zero
       
    4. **Symmetry**:
       - Observe how the gradient field reflects the symmetry of the function
       - For example, in x² + y², the field has radial symmetry
    """)

    add_navigation_buttons(prev_page="Gradient Explanation", next_page="Gradient Applications")

elif st.session_state["current_page"] == "Gradient Applications":
    st.title("Applications of Gradients")
    
    # Create tabs for different application areas
    ml_tab, physics_tab, math_tab = st.tabs(["Machine Learning", "Physics", "Mathematics"])
    
    with ml_tab:
        st.markdown("""
        ## Gradient Descent in Machine Learning
        
        ### What is Gradient Descent?
        Gradient descent is an optimization algorithm that uses gradients to find the minimum of a function. 
        In machine learning, we use it to minimize the error (loss) of our models.
        
        ### How it Works
        1. Start at a random point
        2. Calculate the gradient (direction of steepest increase)
        3. Move in the opposite direction of the gradient
        4. Repeat until reaching a minimum
        
        $$
        \\text{New Position} = \\text{Current Position} - \\text{Learning Rate} \\times \\nabla f
        $$
        
        ### Applications in Machine Learning
        - **Neural Networks**: Training deep learning models
        - **Linear Regression**: Finding optimal coefficients
        - **Logistic Regression**: Optimizing classification models
        - **Support Vector Machines**: Finding optimal decision boundaries
        
        ### Types of Gradient Descent
        1. **Batch Gradient Descent**: Uses entire dataset
        2. **Stochastic Gradient Descent (SGD)**: Uses one sample at a time
        3. **Mini-batch Gradient Descent**: Uses small batches of data
        
        ### Advanced Variants
        - Adam Optimizer
        - RMSprop
        - Momentum
        - AdaGrad
        
        Each variant improves upon basic gradient descent by addressing specific challenges like:
        - Learning rate adaptation
        - Escaping local minima
        - Handling sparse gradients
        """)
        
        # Add interactive visualization for gradient descent
        st.markdown("""
        ### Interactive Visualization: Gradient Descent in 2D
        Watch how gradient descent finds the minimum of a function:
        """)
        
        # Create contour plot of a simple function
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2  # Simple bowl-shaped function
        
        # Create animation frames for gradient descent
        steps = 10
        learning_rate = 0.3
        start_point = np.array([1.5, 1.5])
        points = [start_point]
        
        for _ in range(steps):
            current = points[-1]
            gradient = np.array([2*current[0], 2*current[1]])  # Gradient of x^2 + y^2
            next_point = current - learning_rate * gradient
            points.append(next_point)
        
        points = np.array(points)
        
        # Plot
        fig = go.Figure()
        
        # Add contour plot
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z,
            colorscale='Viridis',
            showscale=True,
            name='Loss Surface'
        ))
        
        # Add gradient descent path
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines+markers',
            name='Gradient Descent Path',
            line=dict(color='red', width=2),
            marker=dict(size=8, symbol='circle')
        ))
        
        fig.update_layout(
            title='Gradient Descent Optimization',
            width=700, height=500,
            xaxis_title='Parameter 1',
            yaxis_title='Parameter 2'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with physics_tab:
        st.markdown("""
        ## Applications in Physics
        
        ### Potential Fields
        Gradients are fundamental in understanding potential fields in physics:
        
        1. **Gravitational Fields**
        - The gradient of gravitational potential gives the gravitational field
        - $$\\vec{g} = -\\nabla \\phi$$
        - Where $$\\phi$$ is the gravitational potential
        
        2. **Electric Fields**
        - Electric field is the negative gradient of electric potential
        - $$\\vec{E} = -\\nabla V$$
        - Where V is the electric potential
        
        3. **Magnetic Fields**
        - Magnetic vector potential and its relationship to magnetic fields
        - Used in electromagnetic theory
        
        ### Fluid Dynamics
        - **Pressure Gradients**: Drive fluid flow
        - **Temperature Gradients**: Cause heat flow
        - **Concentration Gradients**: Drive diffusion
        
        ### Quantum Mechanics
        - **Probability Current**: Related to the gradient of wave function phase
        - **Momentum Operator**: Proportional to gradient operator
        - $$\\hat{p} = -i\\hbar\\nabla$$
        """)
        
    with math_tab:
        st.markdown("""
        ## Mathematical Applications
        
        ### Optimization Problems
        1. **Finding Extrema**
        - Local maxima and minima occur where $$\\nabla f = 0$$
        - Second derivatives determine nature of critical points
        
        2. **Constrained Optimization**
        - Method of Lagrange multipliers
        - $$\\nabla f = \\lambda \\nabla g$$
        - Where g is the constraint function
        
        ### Differential Geometry
        1. **Surface Normal**
        - Gradient gives direction perpendicular to level curves
        - Used in computer graphics for shading
        
        2. **Tangent Spaces**
        - Gradient helps define tangent planes to surfaces
        
        ### Vector Calculus
        1. **Conservative Fields**
        - Field F is conservative if $$F = \\nabla f$$ for some scalar function f
        - Important in physics for force fields
        
        2. **Directional Derivatives**
        - Rate of change in any direction
        - $$\\nabla_v f = \\nabla f \\cdot \\vec{v}$$
        
        ### Applications in Analysis
        1. **Maximum Rate of Change**
        - Gradient points in direction of steepest increase
        - Magnitude gives the rate of change
        
        2. **Level Sets**
        - Gradient is perpendicular to level sets
        - Used in image processing and computer vision
        """)
    
    add_navigation_buttons(prev_page="Gradient Visualization", next_page="Quiz")

elif st.session_state["current_page"] == "Quiz":
    st.title("📝 Gradient Learning Module Quiz")
    st.markdown("""
    ### Test Your Understanding
    This quiz covers all the material from the lesson. Try to answer all questions to test your understanding of gradients.
    """)
    
    # Initialize quiz score in session state if not exists
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    
    # Quiz questions and answers
    questions = [
        {
            "question": "What does the gradient of a function represent?",
            "options": [
                "The second derivative of the function",
                "The direction of steepest descent",
                "The direction and magnitude of steepest increase",
                "The average rate of change"
            ],
            "correct": 2,
            "explanation": "The gradient is a vector that points in the direction of steepest increase, with its magnitude indicating how steep that increase is."
        },
        {
            "question": "For a function f(x,y), what is the correct representation of its gradient?",
            "options": [
                "∇f = f'(x)",
                "∇f = (∂f/∂x, ∂f/∂y)",
                "∇f = ∂²f/∂x∂y",
                "∇f = ∫f dx dy"
            ],
            "correct": 1,
            "explanation": "The gradient of a two-variable function is a vector of its partial derivatives with respect to each variable."
        },
        {
            "question": "In machine learning, why do we move in the opposite direction of the gradient during gradient descent?",
            "options": [
                "To increase the loss function",
                "To find the maximum value",
                "To minimize the loss function",
                "To maintain constant error"
            ],
            "correct": 2,
            "explanation": "In gradient descent, we move opposite to the gradient to find the minimum of the loss function, as the gradient points in the direction of steepest increase."
        },
        {
            "question": "What is the relationship between gradient and level curves?",
            "options": [
                "They are parallel to each other",
                "They are perpendicular to each other",
                "They are the same thing",
                "There is no relationship"
            ],
            "correct": 1,
            "explanation": "The gradient vector at any point is perpendicular to the level curve passing through that point."
        },
        {
            "question": "In physics, what does the negative gradient of electric potential (V) give us?",
            "options": [
                "Electric current",
                "Electric resistance",
                "Electric field (E)",
                "Electric charge"
            ],
            "correct": 2,
            "explanation": "The electric field E is equal to the negative gradient of the electric potential V: E = -∇V."
        },
        {
            "question": "What happens to the gradient at a local minimum?",
            "options": [
                "It becomes infinite",
                "It becomes zero",
                "It points upward",
                "It points downward"
            ],
            "correct": 1,
            "explanation": "At a local minimum, the gradient becomes zero as there is no direction of steepest increase."
        },
        {
            "question": "Which of the following is NOT a type of gradient descent?",
            "options": [
                "Batch Gradient Descent",
                "Stochastic Gradient Descent",
                "Linear Gradient Descent",
                "Mini-batch Gradient Descent"
            ],
            "correct": 2,
            "explanation": "Linear Gradient Descent is not a type of gradient descent. The main types are Batch, Stochastic, and Mini-batch Gradient Descent."
        },
        {
            "question": "What is the gradient of f(x,y) = x² + y²?",
            "options": [
                "(x, y)",
                "(2x, 2y)",
                "(2, 2)",
                "(x², y²)"
            ],
            "correct": 1,
            "explanation": "Taking partial derivatives: ∂f/∂x = 2x and ∂f/∂y = 2y, so ∇f = (2x, 2y)."
        },
        {
            "question": "In quantum mechanics, what is the relationship between the momentum operator and the gradient?",
            "options": [
                "p̂ = ∇",
                "p̂ = -iℏ∇",
                "p̂ = ℏ∇",
                "p̂ = i∇"
            ],
            "correct": 1,
            "explanation": "In quantum mechanics, the momentum operator is given by p̂ = -iℏ∇, where ℏ is the reduced Planck constant."
        },
        {
            "question": "What is a conservative field in vector calculus?",
            "options": [
                "A field that conserves energy",
                "A field that points towards the center",
                "A field that can be expressed as the gradient of a scalar function",
                "A field that has constant magnitude"
            ],
            "correct": 2,
            "explanation": "A conservative field is one that can be expressed as the gradient of a scalar potential function."
        }
    ]
    
    # Display questions
    if not st.session_state.quiz_submitted:
        st.session_state.user_answers = {}
        
        for i, q in enumerate(questions):
            st.markdown(f"### Question {i+1}")
            st.markdown(q["question"])
            st.session_state.user_answers[i] = st.radio(
                "Select your answer:",
                q["options"],
                key=f"q_{i}",
                index=None
            )
        
        # Submit button
        if st.button("Submit Quiz"):
            # Calculate score
            score = 0
            for i, q in enumerate(questions):
                if st.session_state.user_answers[i] == q["options"][q["correct"]]:
                    score += 1
            
            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True
            st.rerun()
    
    else:
        # Display results
        score = st.session_state.quiz_score
        st.markdown(f"### Your Score: {score}/10")
        
        # Display percentage and appropriate message
        percentage = (score / 10) * 100
        if percentage >= 90:
            st.success("🌟 Excellent! You have a strong understanding of gradients!")
        elif percentage >= 70:
            st.success("👍 Good job! You understand most concepts well.")
        elif percentage >= 50:
            st.warning("📚 You're on the right track, but might want to review some concepts.")
        else:
            st.error("💪 Keep studying! Try reviewing the material and attempt the quiz again.")
        
        # Review answers
        st.markdown("### Review Your Answers")
        for i, q in enumerate(questions):
            st.markdown(f"#### Question {i+1}")
            st.markdown(q["question"])
            user_answer = st.session_state.user_answers[i]
            correct_answer = q["options"][q["correct"]]
            
            if user_answer == correct_answer:
                st.success(f"✅ Your answer: {user_answer}")
            else:
                st.error(f"❌ Your answer: {user_answer}")
                st.success(f"Correct answer: {correct_answer}")
            
            st.info(f"Explanation: {q['explanation']}")
        
        # Retry button
        if st.button("Try Again"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = 0
            st.rerun()
    
    add_navigation_buttons(prev_page="Gradient Applications")
