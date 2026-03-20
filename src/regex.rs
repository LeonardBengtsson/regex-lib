use crate::math::{BitMatrix, BitVector, NfaVector};
use crate::regex::graph::{Graph, NodeRef};
use crate::regex::parse::{Atom, ConcatExpr, RegexAst};
use crate::utf8::{UnicodeCodepoint, Utf8DecodeError};
use parsable::Parsable;
use std::collections::HashMap;

mod graph;
mod parse;

pub struct Regex {
    token_matrices: HashMap<UnicodeCodepoint, BitMatrix>,
    final_nodes: BitVector,
}

#[derive(Debug, thiserror::Error)]
pub enum RegexParseError {
    #[error("parse error: 'expected regular expression'")]
    MissingParseResultError,
    #[error(
        "parse error at index {}: 'expected {}'",
        .0.first().map_or(0, |e| e.source_position),
        .0.first().map_or("", |e| &e.error[..]),
    )]
    ParseError(parsable::ParseErrorStack),
}

#[derive(Debug, thiserror::Error)]
pub enum RegexError {
    #[error("{0}")]
    ParseError(RegexParseError),
    #[error("invalid utf8 codepoint: {0}")]
    Utf8DecodeError(Utf8DecodeError),
}

impl Regex {
    pub fn new_from_str(source: &str) -> Result<Regex, RegexParseError> {
        Regex::new(source.as_bytes()).map_err(|e| match e {
            RegexError::ParseError(e) => e,
            RegexError::Utf8DecodeError(_) => panic!(
                "valid UTF-8 string shouldn't result in UTF-8 decoding error"
            ),
        })
    }

    pub fn new(source: &[u8]) -> Result<Regex, RegexError> {
        let mut stream = parsable::ScopedStream::new(source);
        let outcome = RegexAst::parse(&mut stream);
        let regex = match outcome {
            None => {
                return Err(RegexError::ParseError(
                    RegexParseError::MissingParseResultError,
                ));
            }
            Some(result) => match result {
                Ok(regex) => regex,
                Err(e) => {
                    return Err(RegexError::ParseError(
                        RegexParseError::ParseError(e),
                    ));
                }
            },
        };

        let mut graph = Graph::new();
        let start_node = graph.get_initial_node();
        let final_node = graph.add_node();
        graph.set_final(final_node);

        for a in regex.root.node.alts.nodes {
            add_alt(&mut graph, start_node, final_node, a)
                .map_err(RegexError::Utf8DecodeError)?;
        }

        graph.collapse_epsilons();

        let (token_matrices, final_nodes) = graph.compile();

        Ok(Regex {
            token_matrices,
            final_nodes,
        })
    }

    /// returns: whether the entire string matches the regex
    pub fn test(&self, string: &[UnicodeCodepoint]) -> bool {
        let mut accumulator = BitVector::new(self.final_nodes.size);
        // start node
        accumulator.set(0, true);

        let mut temp = BitVector::new(accumulator.size);

        for token in string {
            let Some(matrix) = self.token_matrices.get(token) else {
                return false;
            };
            BitVector::mult(matrix, &accumulator, &mut temp);
            std::mem::swap(&mut accumulator, &mut temp);
        }

        BitVector::dot(&accumulator, &self.final_nodes)
    }

    /// returns: the starting index and length of the first match, if any
    pub fn find(&self, string: &[UnicodeCodepoint]) -> Option<(usize, usize)> {
        let mut accumulator = NfaVector::new(self.final_nodes.size);
        let mut temp = NfaVector::new(accumulator.size);

        // special case for initial final node
        accumulator.set(0, Some(0));
        if NfaVector::dot(&accumulator, &self.final_nodes).is_some() {
            return Some((0, 0));
        }

        let mut earliest_match = None;

        for (token, index) in string.iter().zip(0_usize..) {
            if accumulator.get(0).is_none() {
                accumulator.set(0, Some(index));
            }

            let Some(matrix) = self.token_matrices.get(token) else {
                accumulator.reset();
                continue;
            };
            NfaVector::mult(matrix, &accumulator, &mut temp);
            std::mem::swap(&mut accumulator, &mut temp);

            if let Some(match_index) =
                NfaVector::dot(&accumulator, &self.final_nodes)
            {
                let current_match =
                    Some((match_index, index - match_index + 1));
                if let Some((earliest_match_index, _)) = earliest_match {
                    if match_index < earliest_match_index {
                        earliest_match = current_match;
                    }
                } else {
                    earliest_match = current_match;
                }
            }

            if let Some((earliest_match_index, _)) = earliest_match {
                if !accumulator.iter().any(|i| {
                    i.is_some_and(|index| index < earliest_match_index)
                }) {
                    break;
                }
            }
        }
        earliest_match
    }

    /// returns: the starting index and length of all matches
    pub fn find_all(&self, string: &[UnicodeCodepoint]) -> Vec<(usize, usize)> {
        let mut matches = Vec::new();

        let mut accumulator = NfaVector::new(self.final_nodes.size);
        let mut temp = NfaVector::new(accumulator.size);

        // special case for initial final node
        accumulator.set(0, Some(0));
        if NfaVector::dot(&accumulator, &self.final_nodes).is_some() {
            matches.push((0, 0));
        }

        for (token, index) in string.iter().zip(0_usize..) {
            accumulator.set(0, Some(index));

            let Some(matrix) = self.token_matrices.get(token) else {
                accumulator.reset();
                continue;
            };
            NfaVector::mult(matrix, &accumulator, &mut temp);
            std::mem::swap(&mut accumulator, &mut temp);

            if let Some(start_index) =
                NfaVector::dot(&accumulator, &self.final_nodes)
            {
                matches.push((start_index, index - start_index + 1));
            }
        }
        matches
    }
}

fn add_alt(
    graph: &mut Graph,
    start: NodeRef,
    end: NodeRef,
    alt: ConcatExpr,
) -> Result<(), Utf8DecodeError> {
    let mut prev = start;
    for p in alt.parts.nodes {
        let is_kleene = p.star.is_some();
        let next = if is_kleene { prev } else { graph.add_node() };
        match p.atom {
            Atom::CharacterAtom(c) => {
                let token = c.to_codepoint()?;
                graph.connect(prev, next, token);
            }
            Atom::Capture { alt, .. } => {
                for a in alt.alts.nodes {
                    add_alt(graph, prev, next, a)?;
                }
            }
        }
        prev = next;
    }
    if prev != end {
        graph.connect_epsilon(prev, end);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utf8;

    #[test]
    fn regex_test() {
        fn test(r: &str, s: &str) -> bool {
            Regex::new(r.as_bytes())
                .unwrap()
                .test(&utf8::decode_utf8(s.as_bytes()).unwrap())
        }

        assert!(test("", ""));
        assert!(!test("", "a"));

        assert!(!test("a", "ab"));
        assert!(!test("ab", "a"));

        assert!(test("a(a(b|cd)*|ab)*c", "ac"));
        assert!(test("a(a(b|cd)*|ab)*c", "aac"));
        assert!(test("a(a(b|cd)*|ab)*c", "aabbbbabc"));
        assert!(test("a(a(b|cd)*|ab)*c", "aabbabacdcdabc"));

        assert!(!test("a(a(b|cd)*|ab)*c", ""));
        assert!(!test("a(a(b|cd)*|ab)*c", "a"));
        assert!(!test("a(a(b|cd)*|ab)*c", "c"));
    }

    #[test]
    fn regex_find() {
        fn find(r: &str, s: &str) -> Option<(usize, usize)> {
            Regex::new(r.as_bytes())
                .unwrap()
                .find(&utf8::decode_utf8(s.as_bytes()).unwrap())
        }

        assert_eq!(find("", ""), Some((0, 0)));
        assert_eq!(find("", "a"), Some((0, 0)));

        assert_eq!(find("a", "ab"), Some((0, 1)));
        assert_eq!(find("ab", "a"), None);

        assert_eq!(find("a(a(b|cd)*|ab)*c", "ac"), Some((0, 2)));
        assert_eq!(find("a(a(b|cd)*|ab)*c", "aac"), Some((0, 3)));
        assert_eq!(find("a(a(b|cd)*|ab)*c", "aabbbbabc"), Some((0, 9)));
        assert_eq!(find("a(a(b|cd)*|ab)*c", "aabbabacdcdabc"), Some((0, 8)));

        assert_eq!(find("a(a(b|cd)*|ab)*c", ""), None);
        assert_eq!(find("a(a(b|cd)*|ab)*c", "a"), None);
        assert_eq!(find("a(a(b|cd)*|ab)*c", "c"), None);

        assert_eq!(find("(a|bc)*(c|db)", "abcbcdcadb"), Some((2, 1)));
        assert_eq!(find("(a|bc)*db", "abcbcdcadb"), Some((7, 3)));

        assert_eq!(find("aba|b", "aba"), Some((0, 3)));
        assert_eq!(find("abb*|b", "abbba"), Some((0, 2)));

        assert_eq!(find("A*aB*|D", "AAAa||||b\\"), Some((0, 4)));
        assert_eq!(find("🔥*a\\|*|\\\\", "🔥🔥🔥a||||b\\"), Some((0, 4)));
        assert_eq!(find("🔥*a\\|*b|\\\\", "🔥🔥🔥a||||b\\"), Some((0, 9)));
        assert_eq!(find("🔥*a\\|*|\\\\", "🔥🔥🔥||||b\\"), Some((8, 1)));

        assert_eq!(find("ab", "acab"), Some((2, 2)));
    }
}
