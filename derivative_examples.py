# Dictionary of derivative rules with explanations
DERIVATIVE_RULES = {
    "Power Rule": {
        "rule": "\\frac{d}{dx} x^n = nx^{n-1}",
        "examples": [
            {"input": "x^3", "output": "3x^2"},
            {"input": "x^{1/2}", "output": "\\frac{1}{2}x^{-1/2}"},
            {"input": "5x^4", "output": "20x^3"}
        ]
    },
    "Product Rule": {
        "rule": "\\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)",
        "examples": [
            {"input": "x^2 \\sin(x)", "output": "2x\\sin(x) + x^2\\cos(x)"},
            {"input": "xe^x", "output": "e^x + xe^x"}
        ]
    },
    "Quotient Rule": {
        "rule": "\\frac{d}{dx}\\frac{f(x)}{g(x)} = \\frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}",
        "examples": [
            {"input": "\\frac{x^2}{\\sin(x)}", "output": "\\frac{2x\\sin(x) - x^2\\cos(x)}{\\sin^2(x)}"}
        ]
    },
    "Chain Rule": {
        "rule": "\\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)",
        "examples": [
            {"input": "\\sin(x^2)", "output": "2x\\cos(x^2)"},
            {"input": "e^{x^2}", "output": "2xe^{x^2}"}
        ]
    }
}

# Practice problems
PRACTICE_PROBLEMS = [
    {
        "function": "x^2\\sin(x)",
        "solution": "2x\\sin(x) + x^2\\cos(x)",
        "explanation": "Use the product rule: d/dx[u·v] = u'v + uv' where:\n- u = x² → u' = 2x\n- v = sin(x) → v' = cos(x)"
    },
    {
        "function": "\\frac{x^2 + 1}{x}",
        "solution": "\\frac{2x^2 - (x^2 + 1)}{x^2} = \\frac{x^2 - 1}{x^2}",
        "explanation": "Use the quotient rule: d/dx[u/v] = (u'v - uv')/v² where:\n- u = x² + 1 → u' = 2x\n- v = x → v' = 1"
    },
    {
        "function": "e^{x^2}",
        "solution": "2xe^{x^2}",
        "explanation": "Use the chain rule: d/dx[e^u] = e^u·du/dx where:\n- u = x² → du/dx = 2x"
    },
    {
        "function": "\\sqrt{x^3 + 1}",
        "solution": "\\frac{3x^2}{2\\sqrt{x^3 + 1}}",
        "explanation": "Use the chain rule: d/dx[√u] = 1/(2√u)·du/dx where:\n- u = x³ + 1 → du/dx = 3x²"
    },
    {
        "function": "\\ln(x)\\cos(x)",
        "solution": "\\frac{\\cos(x)}{x} - \\ln(x)\\sin(x)",
        "explanation": "Use the product rule: d/dx[u·v] = u'v + uv' where:\n- u = ln(x) → u' = 1/x\n- v = cos(x) → v' = -sin(x)"
    },
    {
        "function": "\\sin^2(x)",
        "solution": "2\\sin(x)\\cos(x)",
        "explanation": "Use the chain rule: d/dx[sin²(x)] = 2sin(x)·d/dx[sin(x)]"
    },
    {
        "function": "x^3e^x",
        "solution": "x^3e^x + 3x^2e^x",
        "explanation": "Use the product rule: d/dx[u·v] = u'v + uv' where:\n- u = x³ → u' = 3x²\n- v = e^x → v' = e^x"
    },
    {
        "function": "\\frac{\\ln(x)}{x}",
        "solution": "\\frac{1-\\ln(x)}{x^2}",
        "explanation": "Use the quotient rule: d/dx[u/v] = (u'v - uv')/v² where:\n- u = ln(x) → u' = 1/x\n- v = x → v' = 1"
    },
    {
        "function": "\\cos(x^2)",
        "solution": "-2x\\sin(x^2)",
        "explanation": "Use the chain rule: d/dx[cos(u)] = -sin(u)·du/dx where:\n- u = x² → du/dx = 2x"
    },
    {
        "function": "\\sqrt{1-x^2}",
        "solution": "-\\frac{x}{\\sqrt{1-x^2}}",
        "explanation": "Use the chain rule: d/dx[√u] = 1/(2√u)·du/dx where:\n- u = 1-x² → du/dx = -2x"
    },
    {
        "function": "(x+1)^3",
        "solution": "3(x+1)^2",
        "explanation": "Use the chain rule: d/dx[(x+1)³] = 3(x+1)²·d/dx[x+1]"
    },
    {
        "function": "x\\ln(x)",
        "solution": "\\ln(x) + 1",
        "explanation": "Use the product rule: d/dx[u·v] = u'v + uv' where:\n- u = x → u' = 1\n- v = ln(x) → v' = 1/x"
    },
    {
        "function": "\\tan(x)",
        "solution": "\\sec^2(x)",
        "explanation": "Use the quotient rule on sin(x)/cos(x) or recall that d/dx[tan(x)] = sec²(x)"
    },
    {
        "function": "e^{\\sin(x)}",
        "solution": "e^{\\sin(x)}\\cos(x)",
        "explanation": "Use the chain rule: d/dx[e^u] = e^u·du/dx where:\n- u = sin(x) → du/dx = cos(x)"
    },
    {
        "function": "\\frac{1}{\\sqrt{x}}",
        "solution": "-\\frac{1}{2x^{3/2}}",
        "explanation": "Rewrite as x^{-1/2} and use the power rule: d/dx[x^n] = nx^{n-1}"
    },
    {
        "function": "x^2e^{-x}",
        "solution": "e^{-x}(2x - x^2)",
        "explanation": "Use the product rule: d/dx[u·v] = u'v + uv' where:\n- u = x² → u' = 2x\n- v = e^{-x} → v' = -e^{-x}"
    },
    {
        "function": "\\ln(x^2 + 1)",
        "solution": "\\frac{2x}{x^2 + 1}",
        "explanation": "Use the chain rule: d/dx[ln(u)] = 1/u·du/dx where:\n- u = x² + 1 → du/dx = 2x"
    },
    {
        "function": "\\sin(x)\\cos(x)",
        "solution": "\\cos^2(x) - \\sin^2(x)",
        "explanation": "Use the product rule: d/dx[u·v] = u'v + uv' where:\n- u = sin(x) → u' = cos(x)\n- v = cos(x) → v' = -sin(x)"
    },
    {
        "function": "\\frac{e^x}{x}",
        "solution": "\\frac{e^x(x-1)}{x^2}",
        "explanation": "Use the quotient rule: d/dx[u/v] = (u'v - uv')/v² where:\n- u = e^x → u' = e^x\n- v = x → v' = 1"
    },
    {
        "function": "\\sqrt{\\tan(x)}",
        "solution": "\\frac{\\sec^2(x)}{2\\sqrt{\\tan(x)}}",
        "explanation": "Use the chain rule: d/dx[√u] = 1/(2√u)·du/dx where:\n- u = tan(x) → du/dx = sec²(x)"
    }
] 