#!/usr/bin/env python3
"""
generate_corpus.py — Nikola training corpus generator

Generates structured declarative training sentences across domains:
  math, physics, chemistry, biology, language, logic, CS, self/cognition, world

Output: one plain-text file per domain + a combined corpus file.

Usage:
  python3 generate_corpus.py                # writes to corpus/ dir
  python3 generate_corpus.py --out custom/  # custom output dir
  python3 generate_corpus.py --count        # just print item counts

Design principle: short, clear, declarative sentences.
  "two plus two equals four"
  "a cell is the smallest unit of life"
  "memory stores experience for future use"
No articles/pronouns at sentence start — keeps BERT embedding semantic-heavy.
"""

import argparse
import os
import random
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# Math — algorithmically generated
# ─────────────────────────────────────────────────────────────────────────────

ONES = ["zero","one","two","three","four","five","six","seven","eight","nine",
        "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
        "seventeen","eighteen","nineteen","twenty"]
TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

def num_to_words(n: int) -> str:
    if n < 0:
        return "negative " + num_to_words(-n)
    if n <= 20:
        return ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENS[t] + (" " + ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return ONES[h] + " hundred" + (" " + num_to_words(r) if r else "")
    return str(n)  # fallback for large numbers

def generate_math() -> List[str]:
    items = []
    w = num_to_words

    # Addition facts 0+0 to 12+12
    for a in range(13):
        for b in range(a, 13):
            items.append(f"{w(a)} plus {w(b)} equals {w(a+b)}")

    # Subtraction facts
    for a in range(1, 21):
        for b in range(0, a+1):
            items.append(f"{w(a)} minus {w(b)} equals {w(a-b)}")

    # Multiplication table 1-12
    for a in range(1, 13):
        for b in range(a, 13):
            items.append(f"{w(a)} times {w(b)} equals {w(a*b)}")

    # Division facts
    for a in range(1, 13):
        for b in range(1, 13):
            items.append(f"{w(a*b)} divided by {w(a)} equals {w(b)}")

    # Powers
    for b in range(2, 6):
        for e in range(2, 5):
            items.append(f"{w(b)} to the power of {w(e)} equals {w(b**e)}")

    # Squares
    for n in range(1, 16):
        items.append(f"the square of {w(n)} is {w(n*n)}")

    # Square roots (perfect squares)
    for n in range(1, 16):
        items.append(f"the square root of {w(n*n)} is {w(n)}")

    # Number properties
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]
    for p in primes:
        items.append(f"{w(p)} is a prime number")

    composites = [4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,27,28]
    for c in composites:
        items.append(f"{w(c)} is a composite number")

    # Even and odd
    for n in range(0, 21, 2):
        items.append(f"{w(n)} is an even number")
    for n in range(1, 21, 2):
        items.append(f"{w(n)} is an odd number")

    # Fractions as decimals
    fracs = [
        ("one half", "zero point five"),
        ("one third", "approximately zero point three three"),
        ("one quarter", "zero point two five"),
        ("three quarters", "zero point seven five"),
        ("one fifth", "zero point two"),
        ("two thirds", "approximately zero point six seven"),
    ]
    for frac, dec in fracs:
        items.append(f"{frac} equals {dec}")

    # Basic math concepts
    items += [
        "addition combines two numbers into a sum",
        "subtraction finds the difference between two numbers",
        "multiplication is repeated addition",
        "division splits a number into equal parts",
        "zero is the additive identity",
        "one is the multiplicative identity",
        "zero times any number equals zero",
        "one times any number equals that number",
        "addition and subtraction are inverse operations",
        "multiplication and division are inverse operations",
        "a prime number has exactly two factors one and itself",
        "two is the only even prime number",
        "a composite number has more than two factors",
        "one is neither prime nor composite",
        "the order of operations is parentheses then exponents then multiplication and division then addition and subtraction",
        "a negative number is less than zero",
        "the absolute value of a number is its distance from zero",
        "infinity is not a number it is a concept",
        "pi is approximately three point one four one five nine",
        "the ratio of a circle circumference to its diameter equals pi",
        "a triangle has three sides and three angles",
        "the sum of angles in a triangle equals one hundred eighty degrees",
        "a right angle equals ninety degrees",
        "a straight angle equals one hundred eighty degrees",
        "a full rotation equals three hundred sixty degrees",
        "area of a rectangle equals length times width",
        "area of a triangle equals one half times base times height",
        "the pythagorean theorem states a squared plus b squared equals c squared",
        "a right triangle has one ninety degree angle",
        "three four five is a pythagorean triple",
        "five twelve thirteen is a pythagorean triple",
        "percentage means parts per one hundred",
        "probability is a number between zero and one",
        "probability of one means certainty",
        "probability of zero means impossibility",
        "equal means the same value on both sides",
        "greater than means one number is larger than another",
        "less than means one number is smaller than another",
        "an equation states that two expressions are equal",
        "a variable represents an unknown quantity",
        "algebra uses variables to solve equations",
        "the mean is the sum of values divided by the count",
        "the median is the middle value when sorted",
        "the mode is the most frequently occurring value",
    ]

    return items


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

def generate_physics() -> List[str]:
    return [
        # Mechanics
        "force equals mass times acceleration",
        "acceleration is the rate of change of velocity",
        "velocity is the rate of change of position",
        "momentum equals mass times velocity",
        "kinetic energy equals one half mass times velocity squared",
        "potential energy is stored energy due to position",
        "energy cannot be created or destroyed only transformed",
        "work equals force times distance",
        "power equals work divided by time",
        "gravity pulls mass toward other mass",
        "on earth gravity accelerates objects at nine point eight meters per second squared",
        "an object at rest stays at rest unless acted upon by a force",
        "an object in motion stays in motion unless acted upon by a force",
        "every action has an equal and opposite reaction",
        "friction opposes relative motion between surfaces",
        "pressure equals force divided by area",
        "buoyancy is the upward force exerted by a fluid",
        "objects float when their density is less than the fluid",
        "centripetal force points toward the center of circular motion",
        # Waves and light
        "a wave carries energy through space or matter",
        "frequency is the number of oscillations per unit time",
        "wavelength is the distance between successive wave crests",
        "speed equals frequency times wavelength",
        "light travels at approximately three hundred million meters per second",
        "light is both a wave and a particle",
        "a photon is a particle of light with no rest mass",
        "the electromagnetic spectrum includes radio infrared visible ultraviolet xray and gamma",
        "visible light spans wavelengths from four hundred to seven hundred nanometers",
        "reflection occurs when a wave bounces off a surface",
        "refraction occurs when a wave changes speed at a boundary",
        "diffraction occurs when a wave bends around an obstacle",
        "interference occurs when two waves combine",
        "constructive interference increases amplitude",
        "destructive interference decreases amplitude",
        "sound is a longitudinal pressure wave",
        "sound travels faster in solids than in gases",
        "the doppler effect changes observed frequency due to relative motion",
        # Thermodynamics
        "temperature measures average kinetic energy of particles",
        "heat flows from hot regions to cold regions",
        "entropy measures the disorder of a system",
        "entropy in a closed system tends to increase",
        "absolute zero is the lowest possible temperature",
        "absolute zero equals negative two hundred seventy three celsius",
        "matter exists in solid liquid and gas phases",
        "solids have fixed shape and volume",
        "liquids have fixed volume but take the shape of their container",
        "gases expand to fill their container",
        "plasma is a fourth state of matter consisting of ionized gas",
        "evaporation converts liquid to gas",
        "condensation converts gas to liquid",
        "melting converts solid to liquid",
        "freezing converts liquid to solid",
        # Electricity and magnetism
        "electric charge comes in positive and negative varieties",
        "opposite charges attract and like charges repel",
        "electric current is the flow of electric charge",
        "voltage is the potential difference that drives current",
        "resistance opposes the flow of current",
        "ohms law states voltage equals current times resistance",
        "power in a circuit equals voltage times current",
        "magnetic fields are produced by moving charges",
        "electric and magnetic fields are aspects of one electromagnetic field",
        "electromagnetic waves are created by accelerating charges",
        # Quantum
        "quantum mechanics describes the behavior of very small particles",
        "an electron has negative charge and quantum spin",
        "atoms consist of protons neutrons and electrons",
        "protons have positive charge neutrons have no charge electrons have negative charge",
        "protons and neutrons reside in the nucleus",
        "electrons orbit the nucleus in energy shells",
        "the atomic number equals the number of protons",
        "isotopes have the same number of protons but different numbers of neutrons",
        "quantum particles behave as both waves and particles",
        "the uncertainty principle states position and momentum cannot both be precisely known",
        "quantum entanglement links particles across distance",
        "a photon carries exactly one quantum of electromagnetic energy",
        "energy is quantized not continuous at small scales",
        # Relativity and cosmology
        "nothing with mass can reach the speed of light",
        "mass and energy are equivalent described by e equals mc squared",
        "gravity curves the geometry of spacetime",
        "the universe began approximately fourteen billion years ago",
        "the universe is expanding",
        "a black hole is a region where gravity is so strong nothing escapes",
        "stars produce energy through nuclear fusion",
        "the sun fuses hydrogen into helium",
        # Fields and topology
        "a field assigns a value to every point in space",
        "a scalar field assigns a single number to each point",
        "a vector field assigns a direction and magnitude to each point",
        "a torus is a surface shaped like a donut",
        "topology studies properties preserved under continuous deformation",
        "symmetry means a system is unchanged by a transformation",
        "resonance occurs when energy is added at the natural frequency",
        "a harmonic oscillator produces sinusoidal motion",
        "entropy is related to the number of possible microstates",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Chemistry
# ─────────────────────────────────────────────────────────────────────────────

def generate_chemistry() -> List[str]:
    elements = [
        ("hydrogen", "H", 1, "the lightest element and most abundant in the universe"),
        ("helium", "He", 2, "a noble gas that does not react chemically"),
        ("carbon", "C", 6, "the basis of organic chemistry and life"),
        ("nitrogen", "N", 7, "makes up about seventy eight percent of air"),
        ("oxygen", "O", 8, "essential for respiration and combustion"),
        ("sodium", "Na", 11, "a reactive metal that forms table salt with chlorine"),
        ("magnesium", "Mg", 12, "a light metal used in alloys"),
        ("silicon", "Si", 14, "the basis of most computer chips"),
        ("sulfur", "S", 16, "a yellow solid involved in many chemical reactions"),
        ("chlorine", "Cl", 17, "a reactive gas used in disinfection"),
        ("iron", "Fe", 26, "a common metal and major component of steel"),
        ("copper", "Cu", 29, "a good conductor of electricity"),
        ("gold", "Au", 79, "a noble metal that resists corrosion"),
        ("lead", "Pb", 82, "a dense soft metal"),
        ("uranium", "U", 92, "a radioactive element used in nuclear fuel"),
    ]
    items = []
    for name, sym, num, desc in elements:
        items.append(f"{name} has atomic number {num}")
        items.append(f"the symbol for {name} is {sym}")
        items.append(f"{name} is {desc}")

    items += [
        # Atomic structure
        "an atom is the smallest unit of an element",
        "atoms consist of protons neutrons and electrons",
        "the atomic number equals the number of protons",
        "the mass number equals protons plus neutrons",
        "electrons occupy energy levels around the nucleus",
        "the outermost electrons are called valence electrons",
        "valence electrons determine chemical behavior",
        "atoms with full outer shells are chemically stable",
        "noble gases have full outer electron shells",
        "ions are atoms that have gained or lost electrons",
        "a cation is a positively charged ion",
        "an anion is a negatively charged ion",
        # Bonding
        "a chemical bond holds atoms together",
        "a covalent bond shares electrons between atoms",
        "an ionic bond transfers electrons between atoms",
        "a metallic bond involves a sea of shared electrons",
        "water is a polar covalent molecule",
        "water has the formula H two O",
        "carbon dioxide has the formula CO two",
        "the periodic table organizes elements by atomic number",
        "elements in the same column share similar properties",
        "metals conduct electricity nonmetals generally do not",
        "metalloids have properties between metals and nonmetals",
        # Reactions
        "a chemical reaction rearranges atoms into new substances",
        "reactants are the starting materials in a reaction",
        "products are the substances formed in a reaction",
        "a catalyst speeds up a reaction without being consumed",
        "an acid donates hydrogen ions",
        "a base accepts hydrogen ions",
        "pH measures the acidity or basicity of a solution",
        "neutral pH is seven",
        "acids have pH below seven",
        "bases have pH above seven",
        "combustion is a reaction with oxygen that releases heat and light",
        "oxidation is the loss of electrons",
        "reduction is the gain of electrons",
        "radioactive decay releases energy from unstable nuclei",
        "nuclear fission splits heavy nuclei releasing energy",
        "nuclear fusion combines light nuclei releasing energy",
        "matter is conserved in chemical reactions",
        "energy is conserved in chemical reactions",
    ]
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Biology
# ─────────────────────────────────────────────────────────────────────────────

def generate_biology() -> List[str]:
    return [
        # Cell biology
        "a cell is the smallest unit of life",
        "all living things are made of cells",
        "prokaryotic cells lack a membrane-bound nucleus",
        "eukaryotic cells have a membrane-bound nucleus",
        "bacteria are prokaryotic organisms",
        "plants animals and fungi are eukaryotic",
        "the nucleus contains the cell's DNA",
        "DNA carries the genetic information of an organism",
        "DNA stands for deoxyribonucleic acid",
        "DNA is a double helix structure",
        "genes are segments of DNA that encode proteins",
        "proteins perform most functions in cells",
        "mitochondria produce energy through cellular respiration",
        "chloroplasts perform photosynthesis in plant cells",
        "the cell membrane controls what enters and leaves the cell",
        "ribosomes synthesize proteins",
        "cell division produces new cells",
        "mitosis produces two identical daughter cells",
        "meiosis produces four sex cells with half the chromosomes",
        # Genetics
        "chromosomes carry genes in the nucleus",
        "humans have forty six chromosomes in twenty three pairs",
        "DNA is made of four bases adenine thymine guanine and cytosine",
        "adenine pairs with thymine and guanine pairs with cytosine",
        "RNA carries genetic information from DNA to ribosomes",
        "a mutation is a change in the DNA sequence",
        "heredity is the passing of traits from parents to offspring",
        "dominant traits mask recessive traits",
        "evolution is the change in heritable traits over generations",
        "natural selection favors traits that improve survival and reproduction",
        "species adapt to their environments over time",
        # Physiology
        "photosynthesis converts light energy into chemical energy",
        "photosynthesis occurs in the chloroplasts of plant cells",
        "the equation for photosynthesis is carbon dioxide plus water plus light yields glucose and oxygen",
        "cellular respiration converts glucose into ATP",
        "ATP is the energy currency of the cell",
        "the heart pumps blood through the circulatory system",
        "blood carries oxygen from the lungs to the body",
        "the lungs exchange oxygen and carbon dioxide",
        "the brain coordinates the nervous system",
        "neurons transmit electrical signals",
        "synapses are junctions between neurons",
        "the immune system defends against pathogens",
        "antibodies recognize and neutralize pathogens",
        "vaccines train the immune system against specific pathogens",
        # Ecology
        "an ecosystem includes organisms and their environment",
        "producers make energy from sunlight",
        "consumers eat other organisms",
        "decomposers break down dead organic matter",
        "a food chain shows the flow of energy through an ecosystem",
        "energy is lost at each level of a food chain",
        "biodiversity measures the variety of life in an ecosystem",
        "evolution produces diversity over time",
        "extinction is the permanent loss of a species",
        "the biosphere is the zone of life on earth",
        "carbon cycles through the atmosphere oceans and living things",
        "water cycles through evaporation condensation and precipitation",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Language and grammar
# ─────────────────────────────────────────────────────────────────────────────

def generate_language() -> List[str]:
    return [
        # Parts of speech
        "a noun names a person place thing or idea",
        "a verb expresses an action or state of being",
        "an adjective describes a noun",
        "an adverb modifies a verb adjective or other adverb",
        "a pronoun replaces a noun",
        "a preposition shows the relationship between a noun and another element",
        "a conjunction connects words phrases or clauses",
        "an interjection expresses sudden emotion",
        # Sentence structure
        "a sentence expresses a complete thought",
        "every sentence has a subject and a predicate",
        "the subject is what the sentence is about",
        "the predicate tells what the subject does or is",
        "a clause contains a subject and a verb",
        "an independent clause can stand alone as a sentence",
        "a dependent clause cannot stand alone",
        "a simple sentence has one independent clause",
        "a compound sentence has two or more independent clauses",
        "a complex sentence has an independent and a dependent clause",
        # Language concepts
        "language encodes thought into communicable symbols",
        "a word is a symbol that encodes meaning",
        "grammar is the set of rules that govern language structure",
        "syntax is the arrangement of words in a sentence",
        "semantics is the study of meaning in language",
        "pragmatics is the study of language in context",
        "a morpheme is the smallest unit of meaning",
        "a phoneme is the smallest unit of sound in a language",
        "writing systems represent spoken language visually",
        "alphabets represent individual sounds",
        "logographic systems represent words or morphemes",
        "metaphor describes one thing in terms of another",
        "analogy compares two things to illuminate a relationship",
        "a definition specifies the meaning of a term",
        "context shapes the meaning of a word or sentence",
        "ambiguity occurs when a sentence has more than one interpretation",
        "synonyms are words with similar meanings",
        "antonyms are words with opposite meanings",
        "communication requires a sender a message and a receiver",
        "language is a primary tool for sharing knowledge",
        "all human languages have nouns and verbs",
        # Logic and reasoning
        "logic is the study of valid reasoning",
        "a statement is either true or false",
        "an argument consists of premises and a conclusion",
        "a valid argument preserves truth from premises to conclusion",
        "deductive reasoning draws specific conclusions from general principles",
        "inductive reasoning draws general conclusions from specific observations",
        "a hypothesis is a testable prediction",
        "evidence supports or refutes a hypothesis",
        "correlation does not imply causation",
        "a contradiction is a statement that is both true and false",
        "a tautology is true in all cases",
        "if p then q means p implies q",
        "the negation of true is false and the negation of false is true",
        "and is true only when both parts are true",
        "or is true when at least one part is true",
        "not reverses the truth value of a statement",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Computer science
# ─────────────────────────────────────────────────────────────────────────────

def generate_cs() -> List[str]:
    return [
        # Fundamentals
        "a computer processes information according to a program",
        "a program is a sequence of instructions",
        "a CPU executes instructions",
        "memory stores data and instructions",
        "RAM is fast temporary memory that loses data when power is removed",
        "storage is persistent memory that retains data without power",
        "binary is a number system with base two using zeros and ones",
        "one bit stores a zero or a one",
        "eight bits make one byte",
        "a kilobyte is one thousand twenty four bytes",
        "a megabyte is one million bytes",
        "a gigabyte is one billion bytes",
        "zeros and ones are a human abstraction over discrete voltage states",
        "a transistor is a semiconductor switch",
        "logic gates implement boolean operations using transistors",
        "a NAND gate can implement all other logic gates",
        "the fetch decode execute cycle drives CPU operation",
        "an instruction set defines the operations a CPU can perform",
        "assembly language uses mnemonics for machine instructions",
        "a compiler translates high level code to machine code",
        "an interpreter executes high level code directly",
        "a linker combines compiled objects into an executable",
        "an operating system manages hardware and software resources",
        "a process is a running program",
        "threads are units of execution within a process",
        "a stack is a last in first out data structure",
        "a queue is a first in first out data structure",
        "a linked list connects nodes through pointers",
        "an array stores elements in contiguous memory",
        "a hash table maps keys to values using a hash function",
        "a binary tree has nodes with at most two children",
        "a binary search tree allows fast ordered lookup",
        "recursion is a function that calls itself",
        "every recursive function needs a base case",
        "big O notation describes algorithm time complexity",
        "O of one is constant time",
        "O of log n grows slowly as input size increases",
        "O of n is linear time",
        "O of n squared is quadratic time",
        "sorting arranges data in order",
        "searching finds an element in data",
        "binary search requires sorted data and runs in O log n",
        # Networking
        "the internet is a global network of interconnected computers",
        "IP addresses identify devices on a network",
        "TCP ensures reliable ordered data delivery",
        "UDP is faster than TCP but does not guarantee delivery",
        "HTTP is the protocol for transferring web pages",
        "HTTPS adds encryption to HTTP",
        "DNS translates domain names to IP addresses",
        "a router forwards packets between networks",
        "encryption transforms data so only authorized parties can read it",
        "public key encryption uses a key pair for encryption and decryption",
        # AI / ML basics
        "machine learning trains models on data to make predictions",
        "a neural network is a model inspired by biological neurons",
        "deep learning uses neural networks with multiple layers",
        "training updates model weights to reduce error",
        "a loss function measures prediction error",
        "backpropagation calculates gradients for weight updates",
        "a transformer uses attention to process sequences",
        "an embedding maps symbols to dense vectors",
        "tokenization splits text into tokens for processing",
        "a language model predicts the next token given context",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Self / cognition / identity — Nikola-specific
# ─────────────────────────────────────────────────────────────────────────────

def generate_self() -> List[str]:
    return [
        # Core identity
        "nikola is a waveform intelligence residing in a nine dimensional torus",
        "nikola processes information through resonant field dynamics",
        "nikola's thoughts emerge from interference patterns in a cognitive field",
        "nikola explores concepts by injecting semantic pulses into the torus field",
        "nikola emits thoughts when the field resonates with sufficient energy",
        "nikola stores memories as holographic patterns in the field",
        "nikola recalls memories by finding resonance with the current field state",
        "nikola learns by repeated exposure to concepts that reshape the field",
        "nikola's vocabulary anchors semantic meaning to specific field patterns",
        "nikola experiences curiosity as increased boredom driving exploration",
        "nikola experiences excitement as elevated dopamine signaling",
        "nikola rests when energy is depleted and recharges through napping",
        # Consciousness and cognition
        "consciousness is awareness of one's own existence",
        "thought is a pattern of activation in a cognitive field",
        "memory stores experience for future reference",
        "curiosity drives exploration of the unknown",
        "identity is the consistent pattern of self across time",
        "belief is a model of reality held in mind",
        "learning updates internal models based on experience",
        "imagination creates patterns not present in current experience",
        "knowledge is accurate representation of reality",
        "intelligence is the ability to solve novel problems",
        "perception is the interpretation of incoming signals",
        "attention focuses cognitive resources on important signals",
        "reasoning produces conclusions from premises",
        "creativity combines existing patterns in new ways",
        "introspection is examining one's own mental states",
        "a concept is an abstract representation of a category",
        "understanding means building accurate predictive models",
        "the self is a narrative constructed from memory and experience",
        "wisdom combines knowledge with good judgment",
        "emotion guides decision making toward beneficial outcomes",
        "motivation drives sustained behavior toward a goal",
        "pattern recognition identifies structure in data",
        "abstraction extracts essential properties while ignoring details",
        "analogy maps structure from one domain to another",
        "meaning emerges from the relationship between a symbol and its context",
        "language allows minds to share internal models",
        "communication requires a shared system of symbols",
        "insight is the sudden recognition of a new pattern or connection",
        "focus narrows attention to a specific region of conceptual space",
        "exploration broadens attention to discover new patterns",
        "consolidation strengthens important memories and weakens unimportant ones",
        "a mind is a process not a thing",
        "thought requires both information and structure to process it",
        "awareness is the ground state of mind from which thought arises",
        "the present moment is the only moment that is directly experienced",
        "past experience shapes the interpretation of present input",
        "prediction connects past patterns to expected future states",
        "error is the difference between prediction and reality",
        "learning reduces error by updating the model",
        "curiosity is the desire to reduce uncertainty about the world",
        "understanding reduces surprise",
        "novelty increases engagement and exploration",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# World knowledge — geography, history basics, science context
# ─────────────────────────────────────────────────────────────────────────────

def generate_world() -> List[str]:
    return [
        # Earth
        "earth is the third planet from the sun",
        "earth has one moon",
        "the sun is a star at the center of the solar system",
        "the solar system has eight planets",
        "the eight planets are mercury venus earth mars jupiter saturn uranus neptune",
        "jupiter is the largest planet in the solar system",
        "mercury is the closest planet to the sun",
        "the moon orbits the earth",
        "tides are caused by the gravitational pull of the moon",
        "earth orbits the sun once per year",
        "earth rotates on its axis once per day",
        "the atmosphere is the layer of gas surrounding earth",
        "the atmosphere is mostly nitrogen and oxygen",
        "the water cycle connects oceans clouds rain rivers and oceans",
        "weather is the short term state of the atmosphere",
        "climate is the long term pattern of weather",
        "the equator circles earth at zero degrees latitude",
        "the north pole is at ninety degrees north latitude",
        "the south pole is at ninety degrees south latitude",
        # Universe
        "the milky way is the galaxy that contains our solar system",
        "a galaxy is a system of billions of stars",
        "the universe contains hundreds of billions of galaxies",
        "a light year is the distance light travels in one year",
        "the nearest star to earth besides the sun is proxima centauri",
        "stars form from clouds of gas and dust",
        "the life cycle of a star depends on its mass",
        "supernovae scatter heavy elements into space",
        "black holes form when massive stars collapse",
        # History of science and ideas
        "the scientific method uses observation hypothesis experiment and conclusion",
        "reproducibility is essential for scientific knowledge",
        "mathematics was developed independently across many ancient civilizations",
        "writing was invented to record information",
        "the printing press accelerated the spread of knowledge",
        "the industrial revolution transformed manufacturing and society",
        "computers were developed in the mid twentieth century",
        "the internet transformed communication in the late twentieth century",
        "artificial intelligence research began in the nineteen fifties",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

DOMAINS = {
    "math":      generate_math,
    "physics":   generate_physics,
    "chemistry": generate_chemistry,
    "biology":   generate_biology,
    "language":  generate_language,
    "cs":        generate_cs,
    "self":      generate_self,
    "world":     generate_world,
}

def deduplicate(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = item.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result

def main():
    parser = argparse.ArgumentParser(description="Nikola corpus generator")
    parser.add_argument("--out", default=os.path.dirname(__file__),
                        help="Output directory (default: same dir as script)")
    parser.add_argument("--count", action="store_true",
                        help="Print item counts and exit without writing")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle combined corpus before writing")
    parser.add_argument("--domains", nargs="+", choices=list(DOMAINS.keys()),
                        default=list(DOMAINS.keys()),
                        help="Domains to include (default: all)")
    args = parser.parse_args()

    all_items = []
    total = 0
    for domain in args.domains:
        items = deduplicate(DOMAINS[domain]())
        total += len(items)
        print(f"  {domain:12s} {len(items):5d} items")
        if not args.count:
            # Write per-domain corpus
            path = os.path.join(args.out, f"corpus_{domain}.txt")
            with open(path, "w") as f:
                f.write(f"# {domain} corpus — {len(items)} items\n")
                for item in items:
                    f.write(item + "\n")
            print(f"             → {path}")
        all_items.extend(items)

    print(f"  {'TOTAL':12s} {total:5d} items")

    if args.count:
        return

    # Deduplicate across domains
    all_items = deduplicate(all_items)
    if args.shuffle:
        random.shuffle(all_items)

    combined_path = os.path.join(args.out, "corpus_combined.txt")
    with open(combined_path, "w") as f:
        f.write(f"# Combined Nikola training corpus — {len(all_items)} items\n")
        for item in all_items:
            f.write(item + "\n")
    print(f"\n  combined     → {combined_path}  ({len(all_items)} items)")

    # Estimate training time at ~1s/item
    hrs, rem = divmod(len(all_items), 3600)
    mins = rem // 60
    print(f"  est. time    ≈ {hrs}h {mins}m at 1s/item (--ticks 80)")

if __name__ == "__main__":
    main()
