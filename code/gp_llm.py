from gp_evo import tree

gp_expression="""
1. Token Definitions
A valid expression is written in postfix notation (Reverse Polish Notation).
Tokens belong to three categories:
OPERAND
OPERATOR
CONSTANT
________________________________________
2. OPERAND and CONSTANT
Operands can only be one of the following:
2.1 constant c from 1 to 20, should be a specific number
2.2 a variable,, which is the past value list, with the index ranging from 0 to n, represents the value from the distant past to the recent past:
Allowed varaibles:
price
close
turnover_rate
turnover_rate_f
volume_ratio
pb
ps
ps_ttm
total_share
float_share
free_share
total_mv
circ_mv

Meaning:
price  → Closing price
close  → Closing price
turnover_rate  → Turnover rate (%)
turnover_rate_f  → Free float turnover rate
volume_ratio  → Volume ratio
pb  → Price-to-book ratio (PB)
ps  → Price-to-sales ratio (PS)
ps_ttm  → Price-to-sales ratio (TTM)
total_share  → Total shares outstanding (10k shares)
float_share  → Float shares (10k shares)
free_share  → Free float shares (10k shares)
total_mv  → Total market capitalization (10k CNY)
circ_mv  → Circulating market capitalization (10k CNY)

2.3 the result of an operator
Example operands:
price
15
________________________________________
3. OPERATOR Sets
Operators must be selected only from the following fixed sets.
These operators cannot be modified or extended.
________________________________________
3.1 Unary Operators (1 operand)
These operators take one operand.
Allowed operators:
abs
sin
cos
sqrt
Meaning:
abs(x)   → absolute value
sin(x)   → sine
cos(x)   → cosine
sqrt(x)  → square root
________________________________________
3.2 Partial Binary Operators (2 operands, second operand must be constant c)
These operators take two operands, but the second operand must be a constant c.
Token form:
x c operator
Allowed operators:
avg
max
min
lag
rsi

xxx  part of code  xxxx

Partial Binary Operators require the second operand to be a constant.
Binary operators require two operands.
Unary operators require one operand.
The resulting expression must form a valid postfix tree with depth not exceed 15.



"""



def crossover_text(p1,p2,child):
    rule1=tree_to_postfix(p1.save()[0])
    rule2=tree_to_postfix(p2.save()[0])
    if child is not None:
        rule3=tree_to_postfix(child.save()[0])
    text="[Parent1: {tree: <"+rule1+">, IC performance: "+str(round(p1.train_ic,5))+", IR Performance: "+str(round(p1.fitness,5))+"},"
    text+="Parent2: {tree: <"+rule2+">, IC performance: "+str(round(p2.train_ic,5))+", IR Performance: "+str(round(p2.fitness,5))+"}"
    if child is None:
        text+=']'
    else:
        text+=",Child: {tree: <"+rule3+">, IC performance: "+str(round(child.train_ic,5))+", IR Performance: "+str(round(child.fitness,5))+"}]"
    return text



def mutation_text(p,child):
    rule=tree_to_postfix(p.save()[0])
    if child is not None:
        rule2=tree_to_postfix(child.save()[0])
    text="[Parent: {tree: <"+rule+">, IC performance: "+str(round(p.train_ic,5))+", IR Performance: "+str(round(p.fitness,5))+"}"
    if child is None:
        text+=']'
    else:
        text+=",Child: {tree: <"+rule2+">, IC performance: "+str(round(child.train_ic,5))+", IR Performance: "+str(round(child.fitness,5))+"}]"
    return text

def generate(kind,file_path,text):

    file=open(file_path,encoding='utf-8')

    system_text="You act as a genetic programming "+kind+" operator. Your task is to learn from the "+kind+""" results of previous generations and generate a new offspring expression tree.
    The evaluation of expressions is based on quantitative finance metrics:
    •	IC (Information Coefficient) measured for each day
    •	IR (Information Ratio) calculated over the whole period
    Evaluation rules:
    •	Larger |IC| is better.
    •	Larger |IR| is better.
    •	The sign of IC should remain stable (it should not frequently flip between positive and negative across days).
    You will be given historical """+kind+" experience from previous generations, as follows:"
    
    system_text+= "".join(file.readlines())
    
    system_text+="Guidelines for generating the offspring is as follows:"+gp_expression+"Your goal is to generate a new offspring expression by performing "+kind+" and leveraging patterns that previously produced better IC and IR results. I will give you parent expression trees, together with their evaluation results:"

    if kind=='crossover':
        system_text+="""[
            Parent1: {tree: <postfix_expression>, IC performance: IR_total, IR performance: IR_total},
            Parent2: {tree: <postfix_expression>, IC performance: IR_total, IR performance: IR_total}
        ]"""
    else:
        system_text+="""[
            Parent: {tree: <postfix_expression>, IC performance: IR_total, IR performance: IR_total}
        ]"""
    
    system_text+="""
        Your task is to generate a new offspring expression tree.The output format MUST be exactly:

        Child:{tree: <postfix_expression>}

        Example:
        Child:{tree: <x y add 5 mul>}

        xxx  part of code  xxxx
       Instead, use prior experiences and structural observations to judge whether an expression is likely to be overfitted, and aim to reduce the probability of generating overfitted individuals in future generations.
    """

    user_text="The input is "+text

    return system_text,user_text
    

def think(kind,file_path,text):
    system_text="You will be provided with the offspring generated by "+kind+" in the current generation and their corresponding performance results. The data will be presented in the following format:"
    if kind=='crossover':
        system_text+="""
        {
        [
            Parent1: {tree: <postfix_expression>, IC performance: IR_total, IR performance: IR_total},
            Parent2: {tree: <postfix_expression>, IC performance: IR_total, IR performance: IR_total},
            Child:  {tree: <postfix_expression>, IC performance: IR_total, IR performance: IR_total}
        ],
        ...
        }
        """
    if kind=='mutation':
        system_text+="""
        {
        [
            Parent: {tree: <postfix_expression>, IC performance: IR_total, IR performance: IR_total},
            Child:  {tree: <postfix_expression>, IC performance: IR_total, IR performance: IR_total}
        ],
        ...
        }
        """
    system_text+="""
    The evaluation metrics are based on quantitative finance:
        •	IC (Information Coefficient) measured for each day
        •	IR (Information Ratio) measured for the whole period
        Better expressions typically satisfy the following properties:
        •	Larger |IC| values across days
        •	Larger |IR| values overall
        •	Stable IC sign, meaning IC should not frequently flip between positive and negative

    You are also given a set of past experiences or heuristics that were previously used to guide the generation of child. Your task is to analyze and reflect on both:
        - The current generation results
        - The past experiences
    
    You should analyze which structural changes improved or worsened the performance, such as:
        •	beneficial operator substitutions
        •	useful sub-expression replacements
        •	effective factor combinations
        •	patterns that increase IC stability
        •	etc.
        
    Specifically:
        1. Analyze the effectiveness of the previous experiences using the current generation results.
        2. Identify which past experiences appear useful, which are ineffective, and which may need revision.
        3. Update or refine existing experiences if necessary.
        4. Generate new experiences or heuristics that could help guide future crossover and mutation operations.
    
    Output a summarized list of experiences that should guide the next generation.

    Requirements:
    - Each experience should be concise but contain enough information to be clearly understood by the model.
    - The number of experiences is not fixed; generate as many as are appropriate.
    - Avoid redundant or overly vague statements.
    - Focus on actionable insights that can improve future """+kind+"""

    Output format:
    Experience 1: ...
    Experience 2: ...
    Experience 3: ...
    ...
"""

    file=open(file_path,encoding='utf-8')
    user_text="The past experiences or heuristics that were previously used to guide the generation of child: "+"".join(file.readlines())
        
    user_text+="The results of generation of child are as follows: "+text

    user_text+="""Your task is to analyze and reflect on both the current generation results and the past experiences, and output a summarized list of experiences to guide the next generation.'
    
    In addition, extremely high IC or IR values should not always be regarded as ideal results, as they may indicate potential overfitting during the training process. Such expressions may capture noise or dataset-specific patterns rather than robust and generalizable signals. Therefore, avoid blindly favoring expressions with the highest IC or IR. Instead, use prior experiences and structural patterns to judge potential overfitting, and formulate experiences that help reduce the probability of generating overfitted individuals in future generations.
    """

    return system_text,user_text
