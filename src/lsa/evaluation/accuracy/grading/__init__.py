"""What turns a reply into a score, where lm-eval's own falls short.

`letters` reads GPQA's answer letter, `mathgrade` the boxed math answer, `longbench` LongBench v2's stated
choice, `graphwalks` GraphWalks' stated node set and `ruler` RULER's official string match.
`tasks` wires each into the lm-eval task it grades.
"""
