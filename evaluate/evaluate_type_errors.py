1 - For transforms that:
    - redefine a column
    - derive a column but have a single input (effectively rename and redefine)
    - take all forward calls that do not depend on anything except that initial dataframe
    and reference the column by name
    - try the call without the transform (but rename if necessary for derives)
    and with the transform and see what fraction produce a type error before
    and after (it would be super useful to use this to fix type errors)!


for extracted_function that redefines/derives from single column:
    forward_calls <- get all forward calls that depend on return value and explicitly use column
    for fwd_call in forward_calls:
            p <- slice(graph, fwd_call)
            p' <- remove(p, extracted_function.nodes)
            if extracted_function derives new columns:
                replace references after occurrence of extracted_function to original
            try:
                execute(p')
            except TypeError:
                type_error +=1
                p'_fixed <- add extracted_function to p'
                try:
                    execute(p_fixed)
                    type_error confirmed
                except TypeError:
                    type_error bad, try something else
