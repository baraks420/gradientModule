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
        1. **Directional Derivative**: The directional derivative in direction v is:
           $$D_v f(x) = \\nabla f(x) \\cdot v$$

        2. **Maximum Rate of Change**: The gradient points in the direction of steepest ascent, with magnitude:
           $$\\|\\nabla f(x)\\| = \\max_{\\|v\\|=1} D_v f(x)$$

        3. **Level Sets**: At any point, the gradient is orthogonal to the level set passing through that point.

        ### Critical Points and Optimization
        - **Critical Points**: Points where ∇f = 0 or ∇f doesn't exist
        - **Classification**:
            * If eigenvalues > 0: Local minimum
            * If eigenvalues < 0: Local maximum
            * If mixed eigenvalues: Saddle point
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
    ## Interactive Gradient Visualization

    Understanding gradients through visual representation is crucial for developing intuition. 
    Below are several interactive visualizations that demonstrate key gradient concepts.
    """)

    # Create tabs for different visualizations
    contour_tab, surface_tab, field_tab = st.tabs(["Contour Plot & Gradient", "3D Surface", "Gradient Field"])

    with contour_tab:
        st.markdown("""
        ### Contour Plot with Gradient Vectors
        
        The contour plot shows level sets of the function, with gradient vectors indicating the direction of steepest ascent.
        Observe how the gradient vectors are perpendicular to the level curves.
        """)

        # Function selection for contour plot
        contour_function = st.selectbox(
            "Select function to visualize:",
            ["f(x,y) = x² + y²", 
             "f(x,y) = x² - y²",
             "f(x,y) = sin(x) + cos(y)",
             "f(x,y) = x*y",
             "f(x,y) = e^(-x² - y²)"],
            key="contour_function"
        )

        # Generate data for contour plot
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)

        # Calculate Z based on selected function
        if contour_function == "f(x,y) = x² + y²":
            Z = X**2 + Y**2
            grad_x = 2*X
            grad_y = 2*Y
        elif contour_function == "f(x,y) = x² - y²":
            Z = X**2 - Y**2
            grad_x = 2*X
            grad_y = -2*Y
        elif contour_function == "f(x,y) = sin(x) + cos(y)":
            Z = np.sin(X) + np.cos(Y)
            grad_x = np.cos(X)
            grad_y = -np.sin(Y)
        elif contour_function == "f(x,y) = x*y":
            Z = X*Y
            grad_x = Y
            grad_y = X
        else:  # f(x,y) = e^(-x² - y²)
            Z = np.exp(-X**2 - Y**2)
            grad_x = -2*X*np.exp(-X**2 - Y**2)
            grad_y = -2*Y*np.exp(-X**2 - Y**2)

        # Create contour plot with gradient vectors
        fig = go.Figure()

        # Add contour plot
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z,
            colorscale='Viridis',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        ))

        # Add gradient vectors (quiver plot)
        skip = 10  # Show fewer arrows for clarity
        quiver_x = X[::skip, ::skip].flatten()
        quiver_y = Y[::skip, ::skip].flatten()
        grad_x_quiver = grad_x[::skip, ::skip].flatten()
        grad_y_quiver = grad_y[::skip, ::skip].flatten()

        # Normalize gradient vectors for better visualization
        magnitudes = np.sqrt(grad_x_quiver**2 + grad_y_quiver**2)
        grad_x_quiver = grad_x_quiver / (magnitudes + 1e-10)
        grad_y_quiver = grad_y_quiver / (magnitudes + 1e-10)

        for i in range(len(quiver_x)):
            fig.add_trace(go.Scatter(
                x=[quiver_x[i], quiver_x[i] + 0.2*grad_x_quiver[i]],
                y=[quiver_y[i], quiver_y[i] + 0.2*grad_y_quiver[i]],
                mode='lines+markers',
                line=dict(color='red', width=1),
                marker=dict(size=2),
                showlegend=False
            ))

        fig.update_layout(
            title="Contour Plot with Gradient Vectors for " + contour_function,
            width=800,
            height=800
        )

        st.plotly_chart(fig)

    with surface_tab:
        st.markdown("""
        ### 3D Surface Plot
        
        The 3D surface plot helps visualize how the gradient relates to the slope of the surface.
        The gradient at any point is the vector of partial derivatives, which geometrically represents the steepest slope.
        """)

        # Function selection for 3D surface
        surface_function = st.selectbox(
            "Select function to visualize:",
            ["f(x,y) = x² + y²", 
             "f(x,y) = x² - y²",
             "f(x,y) = sin(x) + cos(y)",
             "f(x,y) = x*y",
             "f(x,y) = e^(-x² - y²)"],
            key="surface_function"
        )

        # Create 3D surface plot
        fig = go.Figure()

        # Calculate Z based on selected function (reusing previous calculations)
        if surface_function == "f(x,y) = x² + y²":
            Z = X**2 + Y**2
        elif surface_function == "f(x,y) = x² - y²":
            Z = X**2 - Y**2
        elif surface_function == "f(x,y) = sin(x) + cos(y)":
            Z = np.sin(X) + np.cos(Y)
        elif surface_function == "f(x,y) = x*y":
            Z = X*Y
        else:  # f(x,y) = e^(-x² - y²)
            Z = np.exp(-X**2 - Y**2)

        fig.add_trace(go.Surface(x=X, y=Y, z=Z))

        fig.update_layout(
            title="3D Surface Plot of " + surface_function,
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="f(x,y)",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=800
        )

        st.plotly_chart(fig)

    with field_tab:
        st.markdown("""
        ### Gradient Vector Field
        
        The gradient vector field shows how the gradient vectors vary across the domain.
        This visualization helps understand:
        - Direction of steepest ascent at each point
        - Magnitude of the gradient (indicated by arrow length)
        - Critical points where gradient vanishes
        """)

        # Function selection for gradient field
        field_function = st.selectbox(
            "Select function to visualize:",
            ["f(x,y) = x² + y²", 
             "f(x,y) = x² - y²",
             "f(x,y) = sin(x) + cos(y)",
             "f(x,y) = x*y",
             "f(x,y) = e^(-x² - y²)"],
            key="field_function"
        )

        # Create gradient field plot
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)

        # Calculate gradients based on selected function
        if field_function == "f(x,y) = x² + y²":
            grad_x = 2*X
            grad_y = 2*Y
        elif field_function == "f(x,y) = x² - y²":
            grad_x = 2*X
            grad_y = -2*Y
        elif field_function == "f(x,y) = sin(x) + cos(y)":
            grad_x = np.cos(X)
            grad_y = -np.sin(Y)
        elif field_function == "f(x,y) = x*y":
            grad_x = Y
            grad_y = X
        else:  # f(x,y) = e^(-x² - y²)
            grad_x = -2*X*np.exp(-X**2 - Y**2)
            grad_y = -2*Y*np.exp(-X**2 - Y**2)

        fig = go.Figure()

        # Normalize gradients for visualization
        magnitudes = np.sqrt(grad_x**2 + grad_y**2)
        grad_x_norm = grad_x / (magnitudes + 1e-10)
        grad_y_norm = grad_y / (magnitudes + 1e-10)

        # Plot gradient vectors
        for i in range(len(x)):
            for j in range(len(y)):
                fig.add_trace(go.Scatter(
                    x=[X[i,j], X[i,j] + 0.2*grad_x_norm[i,j]],
                    y=[Y[i,j], Y[i,j] + 0.2*grad_y_norm[i,j]],
                    mode='lines+markers',
                    line=dict(color='blue', width=1),
                    marker=dict(size=2),
                    showlegend=False
                ))

        fig.update_layout(
            title="Gradient Vector Field for " + field_function,
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=800,
            showlegend=False
        )

        st.plotly_chart(fig)

    st.markdown("""
    ### Key Observations

    1. **Gradient Direction**
       - Vectors always point in direction of steepest ascent
       - Length indicates the magnitude of the rate of change

    2. **Critical Points**
       - Observe where gradient vectors vanish (zero magnitude)
       - These points represent local maxima, minima, or saddle points

    3. **Level Sets**
       - Gradient vectors are perpendicular to level curves
       - This property is fundamental in optimization algorithms
    """)

    add_navigation_buttons(prev_page="Gradient Explanation", next_page="Gradient Applications")

elif st.session_state["current_page"] == "Gradient Applications":
    st.title("Applications of Gradients in Science and Engineering")
    
    st.markdown("""
    ## Real-World Applications of Gradients

    Gradients are fundamental tools in various fields of science, engineering, and computer science. 
    Let's explore some key applications with interactive examples.
    """)

    # Create tabs for different applications
    ml_tab, physics_tab, optimization_tab = st.tabs(["Machine Learning", "Physics & Engineering", "Optimization Problems"])

    with ml_tab:
        st.markdown("""
        ### Machine Learning Applications
        
        #### 1. Gradient Descent in Neural Networks
        Gradient descent is the cornerstone of neural network training. It helps minimize the loss function by iteratively adjusting weights and biases.
        
        $$
        w_{t+1} = w_t - \\alpha \\nabla L(w_t)
        $$
        
        where:
        - \(w_t\) is the weight vector at step t
        - \(\\alpha\) is the learning rate
        - \(\\nabla L(w_t)\) is the gradient of the loss function
        
        #### 2. Types of Gradient Descent
        - **Batch Gradient Descent**: Uses entire dataset
        - **Stochastic Gradient Descent (SGD)**: Uses single sample
        - **Mini-batch Gradient Descent**: Uses small batch of samples
        
        #### Interactive Gradient Descent Visualization
        """)

        # Simple interactive gradient descent visualization
        def loss_function(x, y):
            return (x - 2)**2 + (y - 1)**2

        def gradient(x, y):
            return np.array([2*(x - 2), 2*(y - 1)])

        # Parameters for gradient descent
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
        iterations = st.slider("Number of Iterations", 1, 50, 20)

        # Initial point
        x0 = st.number_input("Initial x", value=-2.0, step=0.1)
        y0 = st.number_input("Initial y", value=-1.0, step=0.1)

        # Perform gradient descent
        points = [(x0, y0)]
        x, y = x0, y0
        for _ in range(iterations):
            grad = gradient(x, y)
            x = x - learning_rate * grad[0]
            y = y - learning_rate * grad[1]
            points.append((x, y))

        # Create contour plot with gradient descent path
        x_range = np.linspace(-3, 5, 100)
        y_range = np.linspace(-3, 5, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = loss_function(X, Y)

        fig = go.Figure()

        # Add contour plot
        fig.add_trace(go.Contour(
            x=x_range, y=y_range, z=Z,
            colorscale='Viridis',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        ))

        # Add gradient descent path
        points = np.array(points)
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines+markers',
            name='Gradient Descent Path',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title="Gradient Descent Optimization",
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600
        )

        st.plotly_chart(fig)

    with physics_tab:
        st.markdown("""
        ### Physics & Engineering Applications
        
        #### 1. Electromagnetic Fields
        The gradient of electric potential (V) gives the electric field (E):
        
        $$
        \\vec{E} = -\\nabla V
        $$
        
        #### 2. Heat Flow
        Temperature gradients determine heat flow direction:
        
        $$
        \\vec{q} = -k\\nabla T
        $$
        
        where:
        - \(\\vec{q}\) is heat flux
        - k is thermal conductivity
        - \(\\nabla T\) is temperature gradient
        
        #### 3. Fluid Dynamics
        Pressure gradients drive fluid flow:
        
        $$
        \\vec{F} = -\\nabla p
        $$
        
        #### Interactive Heat Flow Simulation
        """)

        # Simple heat flow visualization
        def temperature_field(x, y):
            return 100 * np.exp(-(x**2 + y**2)/4)

        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        T = temperature_field(X, Y)

        # Calculate temperature gradient
        dx, dy = np.gradient(T)
        
        fig = go.Figure()

        # Add temperature contour
        fig.add_trace(go.Contour(
            x=x, y=y, z=T,
            colorscale='Hot',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        ))

        # Add heat flow vectors
        skip = 3
        quiver_x = X[::skip, ::skip].flatten()
        quiver_y = Y[::skip, ::skip].flatten()
        dx_quiver = dx[::skip, ::skip].flatten()
        dy_quiver = dy[::skip, ::skip].flatten()

        # Normalize vectors
        magnitudes = np.sqrt(dx_quiver**2 + dy_quiver**2)
        dx_norm = dx_quiver / (magnitudes + 1e-10)
        dy_norm = dy_quiver / (magnitudes + 1e-10)

        for i in range(len(quiver_x)):
            fig.add_trace(go.Scatter(
                x=[quiver_x[i], quiver_x[i] - 0.2*dx_norm[i]],
                y=[quiver_y[i], quiver_y[i] - 0.2*dy_norm[i]],
                mode='lines',
                line=dict(color='blue', width=1),
                showlegend=False
            ))

        fig.update_layout(
            title="Heat Flow Visualization",
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600
        )

        st.plotly_chart(fig)

    with optimization_tab:
        st.markdown("""
        ### Optimization Applications
        
        #### 1. Constrained Optimization
        Using gradients with Lagrange multipliers:
        
        $$
        \\nabla f(x^*) = \\lambda \\nabla g(x^*)
        $$
        
        #### 2. Portfolio Optimization
        Maximizing returns while minimizing risk:
        
        $$
        \\nabla \\left(\\sum_i w_i\\mu_i - \\lambda\\sum_{i,j} w_iw_j\\sigma_{ij}\\right) = 0
        $$
        
        #### 3. Image Processing
        Edge detection using gradient magnitude:
        
        $$
        ||\\nabla I|| = \\sqrt{\\left(\\frac{\\partial I}{\\partial x}\\right)^2 + \\left(\\frac{\\partial I}{\\partial y}\\right)^2}
        $$
        
        #### Interactive Optimization Example
        """)

        # Simple portfolio optimization visualization
        st.markdown("""
        Consider a two-asset portfolio optimization problem:
        - Asset 1: Expected return = 10%, Risk = 20%
        - Asset 2: Expected return = 6%, Risk = 10%
        
        The optimization goal is to maximize the Sharpe ratio:
        
        $$
        \\text{Sharpe Ratio} = \\frac{\\text{Portfolio Return}}{\\text{Portfolio Risk}}
        $$
        """)

        # Portfolio weights
        w1 = st.slider("Weight of Asset 1", 0.0, 1.0, 0.5, 0.01)
        w2 = 1 - w1

        # Calculate portfolio metrics
        portfolio_return = w1 * 0.10 + w2 * 0.06
        portfolio_risk = np.sqrt(w1**2 * 0.20**2 + w2**2 * 0.10**2)
        sharpe_ratio = portfolio_return / portfolio_risk

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Return", f"{portfolio_return:.2%}")
        with col2:
            st.metric("Portfolio Risk", f"{portfolio_risk:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    st.markdown("""
    ## Summary of Gradient Applications

    ### Key Benefits
    1. **Optimization**: Finding optimal solutions in high-dimensional spaces
    2. **Efficiency**: Fast convergence in well-behaved problems
    3. **Versatility**: Applicable across many fields

    ### Limitations
    1. **Local Optima**: May get stuck in local minima
    2. **Scaling**: Requires careful learning rate selection
    3. **Conditioning**: Performance depends on problem structure

    ### Best Practices
    1. **Normalization**: Scale inputs appropriately
    2. **Momentum**: Use momentum terms for faster convergence
    3. **Adaptive Rates**: Consider adaptive learning rates
    """)
    
    add_navigation_buttons(prev_page="Gradient Visualization", next_page="Quiz")
