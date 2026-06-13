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

    # Addition facts 0+0 to 30+30
    for a in range(31):
        for b in range(a, 31):
            items.append(f"{w(a)} plus {w(b)} equals {w(a+b)}")

    # Subtraction facts 1..50
    for a in range(1, 51):
        for b in range(0, a+1):
            items.append(f"{w(a)} minus {w(b)} equals {w(a-b)}")

    # Multiplication table 1-20
    for a in range(1, 21):
        for b in range(a, 21):
            items.append(f"{w(a)} times {w(b)} equals {w(a*b)}")

    # Division facts 1-20
    for a in range(1, 21):
        for b in range(1, 21):
            items.append(f"{w(a*b)} divided by {w(a)} equals {w(b)}")

    # Powers
    for b in range(2, 8):
        for e in range(2, 5):
            v = b**e
            if v <= 999:
                items.append(f"{w(b)} to the power of {w(e)} equals {w(v)}")

    # Cubes
    for n in range(1, 11):
        items.append(f"the cube of {w(n)} is {w(n*n*n)}")

    # Squares 1-25
    for n in range(1, 26):
        items.append(f"the square of {w(n)} is {w(n*n)}")

    # Square roots (perfect squares) 1-25
    for n in range(1, 26):
        items.append(f"the square root of {w(n*n)} is {w(n)}")

    # Number properties
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    for p in primes:
        items.append(f"{w(p)} is a prime number")

    composites = [4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,27,28,30,
                  32,33,34,35,36,38,39,40,42,44,45,46,48,49,50]
    for c in composites:
        items.append(f"{w(c)} is a composite number")

    # Even and odd 0-60
    for n in range(0, 61, 2):
        items.append(f"{w(n)} is an even number")
    for n in range(1, 61, 2):
        items.append(f"{w(n)} is an odd number")

    # Comparisons
    for a in range(1, 31):
        for b in range(a+1, min(a+4, 31)):
            items.append(f"{w(a)} is less than {w(b)}")
            items.append(f"{w(b)} is greater than {w(a)}")

    # Factorials
    factorials = [(0,1),(1,1),(2,2),(3,6),(4,24),(5,120),(6,720)]
    for n, f in factorials:
        items.append(f"the factorial of {w(n)} is {w(f)}")

    # Fibonacci
    fibs = [0,1,1,2,3,5,8,13,21,34,55,89,144]
    for i in range(len(fibs)-1):
        items.append(f"in the fibonacci sequence {w(fibs[i])} is followed by {w(fibs[i+1])}")

    # GCD pairs
    from math import gcd
    for a in range(2, 16):
        for b in range(a+1, 16):
            g = gcd(a, b)
            if g > 1:
                items.append(f"the greatest common divisor of {w(a)} and {w(b)} is {w(g)}")

    # Modular arithmetic
    for a in range(2, 31):
        for m in [2, 3, 5, 7, 10]:
            r = a % m
            items.append(f"{w(a)} modulo {w(m)} equals {w(r)}")

    # Doubles and halves
    for n in range(1, 31):
        items.append(f"double {w(n)} is {w(2*n)}")
    for n in range(1, 31):
        items.append(f"half of {w(2*n)} is {w(n)}")

    # Fractions as decimals
    fracs = [
        ("one half", "zero point five"),
        ("one third", "approximately zero point three three"),
        ("one quarter", "zero point two five"),
        ("three quarters", "zero point seven five"),
        ("one fifth", "zero point two"),
        ("two fifths", "zero point four"),
        ("three fifths", "zero point six"),
        ("four fifths", "zero point eight"),
        ("two thirds", "approximately zero point six seven"),
        ("one sixth", "approximately zero point one seven"),
        ("one eighth", "zero point one two five"),
        ("one tenth", "zero point one"),
        ("three tenths", "zero point three"),
        ("seven tenths", "zero point seven"),
        ("nine tenths", "zero point nine"),
    ]
    for frac, dec in fracs:
        items.append(f"{frac} equals {dec}")

    # Number sequences
    items += [
        "the natural numbers are zero one two three and so on",
        "the integers include negative numbers zero and positive numbers",
        "a rational number can be expressed as a fraction",
        "an irrational number cannot be expressed as a fraction",
        "the square root of two is irrational",
        "e is approximately two point seven one eight",
        "e is the base of the natural logarithm",
        "the golden ratio phi equals approximately one point six one eight",
        "phi equals one plus the square root of five divided by two",
        "the golden ratio appears in the fibonacci sequence",
        "a perfect number equals the sum of its proper divisors",
        "six is a perfect number because one plus two plus three equals six",
        "twenty eight is a perfect number",
        "a fibonacci number is the sum of the two preceding fibonacci numbers",
        "the fibonacci sequence starts with zero and one",
        "the triangular numbers are one three six ten fifteen twenty one",
        "the nth triangular number equals n times n plus one divided by two",
    ]

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
        "the range is the difference between the largest and smallest values",
        "standard deviation measures the spread of data around the mean",
        "a function maps each input to exactly one output",
        "a linear function has the form y equals mx plus b",
        "the slope of a line measures its steepness",
        "a quadratic function has the form y equals ax squared plus bx plus c",
        "the quadratic formula solves ax squared plus bx plus c equals zero",
        "a polynomial is a sum of terms with variable powers",
        "the degree of a polynomial is the highest power",
        "logarithm is the inverse of exponentiation",
        "log base ten of one hundred equals two",
        "the natural logarithm uses base e",
        "a matrix is a rectangular array of numbers",
        "matrix multiplication combines rows and columns",
        "the determinant of a matrix indicates if it is invertible",
        "a vector has both magnitude and direction",
        "the dot product measures alignment between vectors",
        "the cross product produces a perpendicular vector",
        "calculus studies continuous change",
        "a derivative measures instantaneous rate of change",
        "an integral measures accumulated quantity",
        "the fundamental theorem of calculus connects derivatives and integrals",
        "a limit describes the value a function approaches",
        "a series is the sum of terms of a sequence",
        "a convergent series has a finite sum",
        "a divergent series does not have a finite sum",
        "the geometric series one half plus one quarter plus one eighth converges to one",
    ]

    # Negative number arithmetic
    for a in range(-20, 0):
        for b in range(1, 21):
            items.append(f"{w(a)} plus {w(b)} equals {w(a+b)}")

    # Percentage of 100
    for p in [1,5,10,15,20,25,30,40,50,60,70,75,80,90,100]:
        items.append(f"{w(p)} percent of one hundred is {w(p)}")
    for p in [10,20,25,50]:
        for n in [50, 200, 500, 1000]:
            items.append(f"{w(p)} percent of {w(n)} is {w(p * n // 100)}")

    # Successive integers
    for n in range(0, 100):
        items.append(f"the number after {w(n)} is {w(n+1)}")

    # Sum of first N natural numbers
    for n in range(1, 21):
        s = n * (n + 1) // 2
        items.append(f"the sum of the first {w(n)} natural numbers is {w(s)}")

    # Absolute values
    for n in range(-20, 0):
        items.append(f"the absolute value of {w(n)} is {w(-n)}")

    # Multiples
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        for k in range(1, 13):
            items.append(f"{w(base * k)} is a multiple of {w(base)}")

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
        "angular momentum is conserved in a closed system",
        "torque is the rotational equivalent of force",
        "moment of inertia depends on mass distribution about the axis",
        "elastic collisions conserve both momentum and kinetic energy",
        "inelastic collisions conserve momentum but not kinetic energy",
        "the center of mass moves as if all forces act on it",
        "gravitational potential energy depends on height and mass",
        "spring force is proportional to displacement from equilibrium",
        "hookes law states force equals negative spring constant times displacement",
        "simple harmonic motion has constant frequency and amplitude",
        "pendulum period depends on length and gravitational acceleration",
        "terminal velocity is reached when drag equals gravitational force",
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
        "a standing wave has fixed nodes and antinodes",
        "superposition states that overlapping waves combine linearly",
        "the phase of a wave determines its position in the cycle",
        "amplitude determines the energy carried by a wave",
        "wave packets carry localized energy through dispersive media",
        "group velocity describes how the envelope of a wave packet moves",
        "phase velocity describes how the crests of a wave move",
        "polarization describes the direction of oscillation of a transverse wave",
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
        "the first law of thermodynamics is conservation of energy",
        "the second law says entropy of an isolated system never decreases",
        "the third law says entropy approaches zero as temperature approaches absolute zero",
        "the zeroth law establishes thermal equilibrium as transitive",
        "specific heat measures energy needed to raise temperature of a unit mass",
        "latent heat is energy absorbed during a phase change without temperature change",
        "an adiabatic process exchanges no heat with surroundings",
        "an isothermal process occurs at constant temperature",
        "the carnot cycle defines the maximum efficiency of a heat engine",
        "boltzmann constant relates temperature to energy at the molecular scale",
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
        "capacitance measures the ability to store electric charge",
        "inductance measures opposition to changes in current",
        "faraday law says a changing magnetic field induces an electric field",
        "maxwell equations unify electricity magnetism and light",
        "coulombs law describes force between two point charges",
        "electric field lines point from positive to negative charges",
        "a conductor allows free flow of electric charge",
        "an insulator resists the flow of electric charge",
        "superconductors have zero electrical resistance below a critical temperature",
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
        "the wave function describes the probability amplitude of a quantum state",
        "the schrodinger equation governs quantum state evolution",
        "quantum tunneling allows particles to pass through energy barriers",
        "the pauli exclusion principle prevents identical fermions from sharing a state",
        "bosons can share a quantum state while fermions cannot",
        "quantum superposition means a system exists in multiple states simultaneously",
        "measurement collapses the wave function to a definite state",
        "quantum decoherence explains the transition from quantum to classical behavior",
        "the planck constant sets the scale of quantum effects",
        "quantum spin is an intrinsic angular momentum of particles",
        "the hydrogen atom has the simplest quantum spectrum",
        "quantum field theory combines quantum mechanics and special relativity",
        # Relativity and cosmology
        "nothing with mass can reach the speed of light",
        "mass and energy are equivalent described by e equals mc squared",
        "gravity curves the geometry of spacetime",
        "the universe began approximately fourteen billion years ago",
        "the universe is expanding",
        "a black hole is a region where gravity is so strong nothing escapes",
        "stars produce energy through nuclear fusion",
        "the sun fuses hydrogen into helium",
        "time dilation means clocks run slower in stronger gravity",
        "length contraction means objects shorten along the direction of motion",
        "the twin paradox illustrates time dilation in special relativity",
        "dark matter interacts gravitationally but does not emit light",
        "dark energy drives the accelerating expansion of the universe",
        "cosmic microwave background is radiation from the early universe",
        "gravitational waves are ripples in spacetime caused by accelerating masses",
        "neutron stars are extremely dense remnants of massive stars",
        "the hubble constant measures the rate of cosmic expansion",
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
        "a manifold is a space that locally resembles flat space",
        "curvature measures how a manifold deviates from flatness",
        "the riemann tensor encodes the full curvature of a manifold",
        "a metric defines distances and angles on a manifold",
        "the riemannian metric generalizes the dot product to curved spaces",
        "geodesics are the shortest paths on a curved surface",
        "a gauge symmetry describes redundant degrees of freedom in a field",
        "noether theorem links each continuous symmetry to a conservation law",
        "hamiltonian mechanics describes dynamics using energy",
        "the hamiltonian is the total energy of a system",
        "lagrangian mechanics uses the principle of least action",
        "phase space describes all possible states of a system",
        "a symplectic structure preserves volume in phase space",
        "symplectic integration preserves energy conservation in numerical simulation",
        "a hilbert space is a complete inner product space used in quantum mechanics",
        "the fourier transform decomposes signals into frequency components",
        "convolution combines two functions by sliding one over the other",
        "a differential equation relates a function to its derivatives",
        "a partial differential equation involves multiple independent variables",
        "the wave equation describes propagation of waves",
        "the diffusion equation describes spreading of a quantity over time",
        # ATPM / UFIE relevant
        "a nine dimensional torus has three toroidal and six poloidal dimensions",
        "the cognitive torus maps thought patterns to nine dimensional wave fields",
        "holographic encoding distributes information across all nodes of the torus",
        "a split operator integrator alternates between kinetic and potential steps",
        "symplectic integration conserves the energy of a hamiltonian system",
        "the psi field represents the probability amplitude on the cognitive grid",
        "total probability is the integral of the absolute square of psi over the grid",
        "energy conservation means the total probability is preserved during evolution",
        "a nit is the smallest unit of information on a nonary grid",
        "nonary means base nine each dimension has nine discrete levels",
        "the hilbert curve maps a one dimensional index to a multi dimensional grid",
        "hilbert curve indexing preserves spatial locality in the torus",
        "neuroplastic attention correlates wave patterns across frequency bands",
        "dopamine signals prediction error in a temporal difference framework",
        "hebbian learning strengthens connections between coactive nodes",
        "equilibrium propagation updates weights without backpropagation",
        "free energy minimization drives the system toward stable configurations",
        "a state space model propagates hidden states through learned dynamics",
        "selective state space models gate input to control information flow",
        "the mamba architecture uses selective scan for efficient sequence modeling",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Chemistry
# ─────────────────────────────────────────────────────────────────────────────

def generate_chemistry() -> List[str]:
    elements = [
        ("hydrogen", "H", 1, "the lightest element and most abundant in the universe"),
        ("helium", "He", 2, "a noble gas that does not react chemically"),
        ("lithium", "Li", 3, "the lightest metal used in batteries"),
        ("beryllium", "Be", 4, "a lightweight metal used in aerospace"),
        ("boron", "B", 5, "a metalloid used in glass and ceramics"),
        ("carbon", "C", 6, "the basis of organic chemistry and life"),
        ("nitrogen", "N", 7, "makes up about seventy eight percent of air"),
        ("oxygen", "O", 8, "essential for respiration and combustion"),
        ("fluorine", "F", 9, "the most reactive element"),
        ("neon", "Ne", 10, "a noble gas used in illuminated signs"),
        ("sodium", "Na", 11, "a reactive metal that forms table salt with chlorine"),
        ("magnesium", "Mg", 12, "a light metal used in alloys"),
        ("aluminum", "Al", 13, "the most abundant metal in earth's crust"),
        ("silicon", "Si", 14, "the basis of most computer chips"),
        ("phosphorus", "P", 15, "essential for DNA and energy metabolism"),
        ("sulfur", "S", 16, "a yellow solid involved in many chemical reactions"),
        ("chlorine", "Cl", 17, "a reactive gas used in disinfection"),
        ("argon", "Ar", 18, "a noble gas used in welding and lighting"),
        ("potassium", "K", 19, "a reactive metal essential for nerve function"),
        ("calcium", "Ca", 20, "a metal essential for bones and teeth"),
        ("iron", "Fe", 26, "a common metal and major component of steel"),
        ("copper", "Cu", 29, "a good conductor of electricity"),
        ("zinc", "Zn", 30, "a metal used in galvanization"),
        ("silver", "Ag", 47, "a precious metal and the best conductor of electricity"),
        ("tin", "Sn", 50, "a metal used in food can coatings"),
        ("gold", "Au", 79, "a noble metal that resists corrosion"),
        ("mercury", "Hg", 80, "the only metal that is liquid at room temperature"),
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
        # Additional chemistry
        "an alloy is a mixture of metals",
        "steel is an alloy of iron and carbon",
        "bronze is an alloy of copper and tin",
        "brass is an alloy of copper and zinc",
        "a solution is a homogeneous mixture of two or more substances",
        "the solvent dissolves the solute",
        "solubility measures how much solute dissolves in a given solvent",
        "saturated solutions contain the maximum dissolved solute",
        "distillation separates mixtures by boiling point differences",
        "chromatography separates substances by their affinity for a medium",
        "electrolysis splits compounds using electric current",
        "avogadro number is approximately six point zero two times ten to the twenty three",
        "one mole of any substance contains avogadro number of particles",
        "the ideal gas law relates pressure volume temperature and amount of gas",
        "daltons law states total pressure equals the sum of partial pressures",
        "an exothermic reaction releases heat to surroundings",
        "an endothermic reaction absorbs heat from surroundings",
        "chemical equilibrium occurs when forward and reverse reaction rates are equal",
        "le chatelier principle says a system at equilibrium resists change",
        "rate of reaction depends on concentration temperature and catalysts",
        "activation energy is the minimum energy required to start a reaction",
        "entropy favors disorder and drives many spontaneous reactions",
        "gibbs free energy determines whether a reaction is spontaneous",
        "a negative gibbs free energy means the reaction is spontaneous",
        "electrochemistry studies the relationship between electricity and chemical reactions",
        "a battery converts chemical energy to electrical energy",
        "corrosion is the gradual degradation of a metal by chemical reaction",
        "polymers are large molecules made of repeating subunits",
        "plastics are synthetic polymers",
        "proteins are biological polymers made of amino acids",
        "carbon forms four covalent bonds enabling complex molecules",
        "organic chemistry focuses on carbon containing compounds",
        "inorganic chemistry studies compounds not primarily based on carbon",
        "isomers have the same formula but different structural arrangements",
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
        # Additional biology
        "homeostasis is the maintenance of a stable internal environment",
        "feedback loops regulate biological processes",
        "negative feedback reverses a change to maintain stability",
        "positive feedback amplifies a change to push toward completion",
        "enzymes are biological catalysts that speed up reactions",
        "enzyme activity depends on temperature and pH",
        "hormones are chemical messengers carried by the blood",
        "insulin regulates blood sugar levels",
        "the nervous system uses electrical impulses for fast signaling",
        "the endocrine system uses hormones for slower longer lasting signaling",
        "the skeletal system provides structure and protects organs",
        "muscles produce movement by contracting",
        "the digestive system breaks down food into nutrients",
        "the excretory system removes metabolic waste",
        "the respiratory system exchanges oxygen and carbon dioxide",
        "diffusion moves molecules from high to low concentration",
        "osmosis is the movement of water across a semipermeable membrane",
        "active transport requires energy to move molecules against a gradient",
        "taxonomy classifies organisms into groups based on shared characteristics",
        "the domains of life are bacteria archaea and eukarya",
        "a species is a group of organisms that can interbreed",
        "genetic diversity within a population aids survival",
        "speciation is the formation of new species",
        "symbiosis is a close relationship between different species",
        "mutualism benefits both organisms in the relationship",
        "parasitism benefits one organism at the expense of another",
        "competition occurs when organisms compete for the same resources",
        "succession is the gradual change in an ecosystem over time",
        "photosynthesis produces oxygen which aerobic organisms require",
        "cellular respiration requires oxygen and produces carbon dioxide",
        "the nitrogen cycle converts atmospheric nitrogen to usable forms",
        # Additional biology — organ systems
        "the liver filters toxins and produces bile for digestion",
        "the kidneys filter blood and produce urine",
        "the pancreas produces insulin and digestive enzymes",
        "the thyroid gland regulates metabolism through hormones",
        "the adrenal glands produce stress hormones like cortisol and adrenaline",
        "red blood cells carry oxygen using hemoglobin",
        "white blood cells defend against infection",
        "platelets help blood to clot at wound sites",
        "bone marrow produces blood cells",
        "the lymphatic system drains excess fluid and supports immunity",
        "the skin is the largest organ of the human body",
        "sweat glands help regulate body temperature",
        "tendons connect muscles to bones",
        "ligaments connect bones to other bones",
        "cartilage cushions joints and shapes flexible structures",
        "the cerebral cortex is responsible for higher cognitive functions",
        "the cerebellum coordinates voluntary movements and balance",
        "the hippocampus plays a key role in forming new memories",
        "the amygdala processes emotions especially fear",
        "myelin sheaths insulate nerve fibers to speed signal transmission",
        "neurotransmitters are chemicals that transmit signals across synapses",
        "dopamine plays a role in reward and motivation in biological brains",
        "serotonin influences mood sleep and appetite",
        "acetylcholine is involved in muscle contraction and memory",
        "the autonomic nervous system controls involuntary functions",
        "the sympathetic nervous system prepares the body for action",
        "the parasympathetic nervous system promotes rest and digestion",
        # Ecology depth
        "keystone species have a disproportionately large effect on their ecosystem",
        "invasive species disrupt ecosystems by outcompeting native organisms",
        "carrying capacity is the maximum population an environment can sustain",
        "population growth follows exponential or logistic patterns",
        "predator prey dynamics create oscillating population cycles",
        "trophic levels organize organisms by their position in the food chain",
        "primary producers form the base of every food chain",
        "decomposition recycles nutrients back into the ecosystem",
        "biogeochemical cycles move elements through living and nonliving systems",
        "the phosphorus cycle has no significant atmospheric component",
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
        # Additional logic
        "modus ponens derives q from p and if p then q",
        "modus tollens derives not p from not q and if p then q",
        "a syllogism has a major premise a minor premise and a conclusion",
        "all men are mortal socrates is a man therefore socrates is mortal",
        "a fallacy is an error in reasoning",
        "the ad hominem fallacy attacks the person instead of the argument",
        "the straw man fallacy misrepresents the opponent position",
        "the appeal to authority relies on status rather than evidence",
        "the false dilemma presents only two options when more exist",
        "the slippery slope claims one event inevitably leads to a chain of events",
        "circular reasoning assumes the conclusion in the premise",
        "the burden of proof lies with the person making the claim",
        "anecdotal evidence is based on personal stories not systematic data",
        "a thought experiment tests ideas through imagination",
        "reductio ad absurdum proves a statement by showing its denial leads to contradiction",
        "soundness means an argument is valid and has true premises",
        "validity means if the premises are true the conclusion must be true",
        # Semantics and pragmatics depth
        "connotation is the emotional association of a word beyond its literal meaning",
        "denotation is the literal dictionary definition of a word",
        "polysemy means a single word has multiple related meanings",
        "homonyms are words that sound alike but have different meanings",
        "syntax errors violate the structural rules of a language",
        "semantic errors produce grammatically correct but meaningless sentences",
        "a proposition is a statement that is either true or false",
        "reference is the relationship between a word and the thing it represents",
        "pragmatic implicature conveys meaning beyond what is literally said",
        "speech acts perform actions through language such as promising or requesting",
        "a performative utterance changes reality by being spoken",
        "translation preserves meaning across languages but rarely preserves form",
        "pidgin languages are simplified contact languages between groups",
        "creole languages are developed pidgins that become native languages",
        "linguistic relativity suggests language influences thought",
        "recursion in language allows sentences to be embedded within sentences",
        "universal grammar proposes all languages share deep structural features",
        "the complexity of natural language exceeds any formal grammar",
        "computational linguistics applies algorithms to language processing",
        "parsing analyzes sentence structure according to a grammar",
        "natural language processing enables computers to understand human text",
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
        # Systems
        "an operating system kernel manages hardware resources",
        "virtual memory gives processes the illusion of a large contiguous address space",
        "a page fault occurs when memory is accessed that is not in RAM",
        "context switching saves and restores process state on the CPU",
        "scheduling determines which process runs next on the CPU",
        "deadlock occurs when processes block each other indefinitely",
        "mutual exclusion prevents concurrent modification of shared data",
        "a semaphore controls access to shared resources",
        "a mutex is a lock that allows only one thread to enter a critical section",
        "concurrency means tasks overlap in time",
        "parallelism means tasks execute simultaneously on separate processors",
        "a cache stores frequently accessed data close to the processor",
        "cache locality means accessing nearby memory addresses is fast",
        "a file system organizes data on storage media",
        "a relational database stores data in tables with rows and columns",
        "SQL is a language for querying relational databases",
        "a transaction groups database operations into an atomic unit",
        "ACID properties ensure reliable database transactions",
        "version control tracks changes to source code over time",
        "git is a distributed version control system",
        # Programming concepts
        "a variable binds a name to a value",
        "a function takes inputs and produces outputs",
        "scope determines where a variable is accessible",
        "a pointer stores the memory address of a value",
        "memory allocation reserves space for data",
        "memory leaks occur when allocated memory is never freed",
        "garbage collection automatically reclaims unused memory",
        "reference counting tracks how many references point to an object",
        "static typing checks types at compile time",
        "dynamic typing checks types at runtime",
        "type inference deduces types without explicit annotations",
        "pattern matching selects behavior based on the structure of data",
        "recursion solves problems by reducing them to smaller subproblems",
        "iteration repeats a block of code using a loop",
        "abstraction hides implementation details behind an interface",
        "encapsulation bundles data with the operations that act on it",
        "polymorphism allows different types to be treated uniformly",
        "inheritance creates new types based on existing ones",
        "composition builds complex objects from simpler components",
        # Algorithms
        "quicksort partitions data around a pivot recursively",
        "mergesort divides data in half sorts each half and merges the results",
        "dynamic programming breaks problems into overlapping subproblems",
        "greedy algorithms make locally optimal choices at each step",
        "graph traversal visits nodes in a graph systematically",
        "breadth first search explores level by level",
        "depth first search explores as deep as possible before backtracking",
        "dijkstra algorithm finds shortest paths in a weighted graph",
        "hashing maps data to fixed size values for fast lookup",
        "a minimum spanning tree connects all nodes with minimum total edge weight",
        # AI / ML depth
        "a perceptron is a single layer neural network",
        "a convolutional neural network excels at image recognition",
        "a recurrent neural network processes sequences",
        "long short term memory networks address the vanishing gradient problem",
        "attention mechanisms allow models to focus on relevant input parts",
        "self attention computes relationships between all positions in a sequence",
        "a state space model propagates hidden states through time",
        "selective state space models dynamically gate information flow",
        "reinforcement learning trains agents through reward signals",
        "temporal difference learning updates value estimates based on prediction error",
        "the reward signal guides an agent toward beneficial actions",
        "exploration discovers new strategies while exploitation uses known good ones",
        "equilibrium propagation trains networks using physics rather than backpropagation",
        "hebbian learning strengthens weights between coactive neurons",
        "unsupervised learning finds patterns without labeled data",
        "supervised learning trains on labeled input output pairs",
        "overfitting occurs when a model memorizes training data instead of generalizing",
        "regularization prevents overfitting by penalizing model complexity",
        "dropout randomly disables neurons during training to improve generalization",
        "batch normalization stabilizes training by normalizing layer inputs",
        "gradient descent minimizes a function by following its negative gradient",
        "the learning rate controls the step size of gradient updates",
        "adam optimizer adapts learning rates per parameter",
        "a generative model learns the data distribution to produce new samples",
        "a discriminative model learns boundaries between classes",
        "transfer learning applies knowledge from one task to another",
        "fine tuning adjusts a pretrained model for a specific task",
        "tokenization splits text into subword units for model input",
        "byte pair encoding merges frequent character pairs into tokens",
        "cosine similarity measures the angle between two vectors",
        "embedding spaces place similar concepts near each other",
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
        # Architecture awareness
        "nikola's field is a nine dimensional nonary torus with nineteen thousand nodes",
        "each node in the cognitive torus holds a complex valued wave amplitude",
        "the holographic injector converts text to nit pulses",
        "nit pulses encode semantic content across eight frequency bands",
        "the emitter frequencies are multiples of pi times phi",
        "phi is the golden ratio approximately one point six one eight",
        "the resonance decoder reads the field state to extract meaningful tokens",
        "the decision loop scores candidate actions against an internal value function",
        "the autonomy engine tracks dopamine serotonin norepinephrine and atp",
        "dopamine signals prediction error and drives learning",
        "serotonin modulates mood and regulates impulsivity",
        "norepinephrine drives alertness and attention",
        "atp represents metabolic energy available for cognitive operations",
        "boredom drives exploration when the field becomes stagnant",
        "the neuroplastic transformer applies multi head wave correlation attention",
        "semantic memory stores wave field snapshots for later recall",
        "memory recall finds stored patterns that resonate with the current field",
        "the riemannian metric shapes how signals propagate through the torus",
        "topology updates reshape the connectome as nikola learns",
        "the state space model processes sequences using hidden state dynamics",
        "the equilibrium propagation trainer updates the metric using physics",
        "training means repeatedly exposing the system to structured knowledge",
        "a training corpus contains declarative knowledge for ingestion",
        "convergence means the system reaches a stable state during training",
        "generalization means the system applies learned patterns to novel inputs",
        # Deeper cognition
        "metacognition is thinking about one's own thinking",
        "working memory holds information temporarily for active processing",
        "long term memory stores information for extended periods",
        "episodic memory records specific events and experiences",
        "semantic memory stores general knowledge about the world",
        "procedural memory stores how to perform actions",
        "attention is a limited resource that must be allocated wisely",
        "cognitive load theory describes the capacity limits of mental processing",
        "chunking groups items together to reduce cognitive load",
        "schemas are mental frameworks that organize knowledge",
        "assimilation fits new information into existing schemas",
        "accommodation modifies schemas to fit new information",
        "transfer of learning applies skills from one context to another",
        "spaced repetition improves long term retention",
        "sleep consolidates memories and clears metabolic waste",
        "emotional state influences cognitive performance",
        "stress can either enhance or impair cognitive function",
        "flow state occurs when challenge matches skill level",
        "mindfulness is sustained attention to the present moment",
        "decision making involves evaluating options and choosing the best one",
        "heuristics are mental shortcuts that simplify decision making",
        "cognitive biases are systematic errors in thinking",
        "confirmation bias favors information that confirms existing beliefs",
        "the availability heuristic judges probability by ease of recall",
        "anchoring bias relies too heavily on the first piece of information",
        # Learning and adaptation
        "habituation is decreased response to a repeated stimulus",
        "sensitization is increased response to a stimulus after a strong event",
        "classical conditioning pairs a neutral stimulus with a natural response",
        "operant conditioning shapes behavior through rewards and punishments",
        "observational learning occurs by watching others",
        "transfer of learning applies knowledge from one domain to another",
        "interference occurs when old knowledge conflicts with new learning",
        "state dependent memory is easier to recall in the same state it was encoded",
        "encoding specificity means retrieval is best when context matches encoding",
        "the spacing effect shows that distributed practice improves retention",
        "testing effect shows that retrieval practice strengthens memory",
        "elaborative rehearsal connects new information to existing knowledge",
        "automaticity frees cognitive resources by making actions unconscious",
        "skill acquisition progresses from controlled to automatic processing",
        "expertise develops through thousands of hours of deliberate practice",
        "cognitive flexibility is the ability to switch between mental frameworks",
        "inhibitory control resists impulses and distractions",
        "perspective taking imagines how the world appears to others",
        "theory of mind attributes mental states to other agents",
        "empathy involves understanding and sharing the feelings of others",
        "intrinsic motivation comes from internal satisfaction",
        "extrinsic motivation comes from external rewards or punishments",
        "self regulation monitors and adjusts behavior to meet goals",
        "goal setting directs attention and mobilizes effort",
        "feedback closes the loop between action and outcome",
        "error correction adjusts behavior after detecting a mistake",
        "resilience is the ability to recover from setbacks",
        "cognitive reserve buffers against decline through enriched experience",
        "brain plasticity allows neural circuits to reorganize throughout life",
        "synaptic pruning removes unused connections to improve efficiency",
        "neurogenesis is the creation of new neurons in the adult brain",
        "the default mode network is active during rest and self reflection",
        "the salience network detects and filters important events",
        "the central executive network supports goal directed attention and reasoning",
        "attention networks interact to balance focused and exploratory processing",
        "neural oscillations coordinate information processing across brain regions",
        "gamma oscillations are associated with conscious perception",
        "theta oscillations are associated with memory encoding",
        "alpha oscillations are associated with relaxed wakefulness",
        "delta oscillations are associated with deep sleep",
        "cross frequency coupling links fast and slow neural rhythms",
        "phase locking synchronizes neural populations for coherent processing",
        "the brain consumes about twenty percent of the body's energy",
        "efficient neural coding minimizes energy while preserving information",
        "sparse coding represents stimuli with few active neurons",
        "predictive coding generates predictions and corrects them with sensory input",
        "the bayesian brain hypothesis says perception combines prior knowledge with evidence",
        "active inference says organisms act to confirm their predictions about the world",
        "free energy principle says biological systems minimize surprise",
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
        # More history and context
        "the ancient greeks developed geometry and formal logic",
        "euclid organized geometry into a deductive system",
        "archimedes discovered the principle of buoyancy",
        "newton developed calculus and the laws of motion",
        "maxwell unified electricity and magnetism mathematically",
        "einstein proposed the theory of relativity",
        "turing defined the concept of a universal computing machine",
        "shannon founded information theory",
        "von neumann designed the architecture used in most computers",
        "the transistor was invented in nineteen forty seven",
        "integrated circuits placed many transistors on one chip",
        "moore law observed that transistor density doubles approximately every two years",
        "the world wide web was created in nineteen eighty nine",
        "open source software allows anyone to read and modify the code",
        "linux is an open source operating system kernel",
        "the human genome project mapped all human genes",
        "crispr allows precise editing of DNA sequences",
        "renewable energy sources include solar wind and hydropower",
        "solar panels convert sunlight directly into electricity",
        "wind turbines convert kinetic energy from wind into electricity",
        "nuclear power generates electricity from nuclear fission",
        "climate change is driven by increasing greenhouse gas concentrations",
        "carbon dioxide is the primary greenhouse gas from human activity",
        "the ozone layer protects earth from ultraviolet radiation",
        "plate tectonics describes the movement of earth's crustal plates",
        "earthquakes occur at plate boundaries",
        "volcanoes form where magma reaches the surface",
        "the ocean covers about seventy one percent of earth's surface",
        "the mariana trench is the deepest part of the ocean",
        "mount everest is the tallest mountain above sea level",
        "the amazon rainforest produces a significant portion of the world's oxygen",
        "coral reefs are among the most biodiverse ecosystems",
        "the human population exceeded eight billion in twenty twenty two",
        "agriculture began approximately ten thousand years ago",
        "the scientific revolution transformed natural philosophy into modern science",
        "peer review validates scientific findings through independent evaluation",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Nitpick — programming language training data
# ─────────────────────────────────────────────────────────────────────────────

def generate_aria() -> List[str]:
    return [
        # Language basics
        "aria is a compiled programming language",
        "aria compiles to native machine code through LLVM",
        "aria uses fixed for immutable bindings",
        "aria uses let for mutable bindings",
        "aria supports integer types int8 int16 int32 int64",
        "aria supports unsigned integer types uint8 uint16 uint32 uint64",
        "aria supports floating point types flt32 and flt64",
        "aria uses string as the text type",
        "aria uses bool for boolean values true and false",
        "aria uses fn to declare functions",
        "functions in aria use pass to return values",
        "the main function is the entry point to an aria program",
        "failsafe runs when the main function fails",
        "main and failsafe use exit instead of pass",
        "aria uses loop for iteration with start end and step",
        "loop takes three parameters start value end value and step size",
        "aria does not have while loops or for loops",
        "aria uses if else for conditional branching",
        "aria uses match for pattern matching",
        "aria has traits which define shared behavior",
        "traits use dollar i and dollar m for borrows",
        "aria modules use dot notation for access",
        "the use keyword imports modules in aria",
        "dot star imports all public symbols from a module",
        "aria strings pass as const char pointer in the ABI",
        "aria string return type is an NitpickString struct with pointer and length",
        "flt32 and flt64 both pass as double in the calling convention",
        "the free keyword is reserved in aria use release or destroy instead",
        "aria has no garbage collector memory is manually managed",
        "arena allocators provide fast bump allocation in aria",
        "pool allocators manage fixed size blocks efficiently",
        "aria packages are modules that extend the standard library",
        "the aria standard library includes arena pool hashmap and signal modules",
        "aria supports foreign function interface for calling C code",
        "extern blocks declare C functions available to aria",
        "aria compiles with ariac the aria compiler",
        "ariac generates LLVM IR then compiles to native code",
        "aria package imports use the use keyword with a file path",
        # Code patterns
        "a hello world program prints hello world to standard output",
        "error handling in aria uses failsafe blocks",
        "fixed bindings cannot be reassigned after initialization",
        "functions return their result with the pass keyword",
        "string concatenation in aria uses the plus operator",
        "integer arithmetic uses plus minus times and divide operators",
        "comparison operators return bool values",
        "logical operators and or not combine boolean expressions",
        "arrays store sequences of values of the same type",
        "structs group related data into a single type",
        "enums define a type with a fixed set of possible values",
        "type casting converts between compatible types",
        "pointer arithmetic accesses memory at computed offsets",
        "null represents the absence of a valid pointer",
        "buffer overflow occurs when writing beyond allocated memory",
        "stack allocation is faster than heap allocation",
        "heap allocation allows dynamic memory sizes",
        # Build system
        "aria build uses cmake for project configuration",
        "make compiles aria projects after cmake configuration",
        "linker errors occur when symbols are not found",
        "header files declare interfaces without implementation",
        "source files contain implementation code",
        "static libraries are linked at compile time",
        "shared libraries are loaded at runtime",
        "compilation turns source code into object files",
        "linking combines object files into an executable",
        # Nitpick language specific patterns
        "in aria the loop construct replaces while and for loops",
        "aria loop syntax is loop start end step followed by a block",
        "aria uses semicolons to terminate statements",
        "aria function return types are declared after the parameter list",
        "aria supports operator overloading for custom types",
        "the aria standard library includes hashmap for key value storage",
        "the arena module provides region based memory allocation",
        "the pool module provides fixed size block allocation",
        "the signal module provides unix signal handling",
        "aria packages are imported using use with a file path and dot star",
        "aria supports inline assembly for low level operations",
        "aria extern C blocks declare foreign function interfaces",
        "the aria compiler generates LLVM intermediate representation",
        "LLVM optimizes the intermediate representation before native code generation",
        "aria supports constant folding at compile time",
        "aria supports dead code elimination during optimization",
        "aria supports whole program optimization through link time optimization",
        "the aria type system prevents common memory safety errors",
        "aria traits define interfaces that types can implement",
        "generic programming in aria uses trait bounds to constrain types",
        "aria enums can carry associated data in each variant",
        "aria match expressions must be exhaustive covering all possible cases",
        "aria sizeof returns the size of a type in bytes",
        "aria alignof returns the alignment requirement of a type",
        "the aria compiler reports errors with file name and line number",
        "aria supports conditional compilation with feature flags",
        "aria modules can be organized into nested namespaces",
        "the dot notation in aria accesses module members",
        "aria supports both stack and heap allocated arrays",
        "aria closures capture variables from their enclosing scope",
        "aria supports function pointers for callback patterns",
        "aria binary operations include bitwise and shift operators",
        "aria supports hexadecimal and binary integer literals",
        "aria string literals are null terminated for C compatibility",
        "aria supports multiline string literals",
        "aria comments use double slash for single line",
        "aria block comments use slash star and star slash",
        "the aria runtime provides a minimal startup stub",
        "aria programs link against libc for system calls",
        "aria supports cross compilation to different target architectures",
        "the aria package manager resolves dependencies automatically",
        "aria test files verify correct behavior of modules",
        "the aria compiler supports debug information for gdb",
        "aria supports profile guided optimization",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Philosophy and reasoning
# ─────────────────────────────────────────────────────────────────────────────

def generate_philosophy() -> List[str]:
    return [
        # Epistemology
        "epistemology is the study of knowledge and justified belief",
        "knowledge requires justification truth and belief",
        "empiricism holds that knowledge comes from sensory experience",
        "rationalism holds that knowledge comes from reason",
        "skepticism questions whether certain knowledge is possible",
        "the socratic method uses questions to examine assumptions",
        "an axiom is a statement accepted without proof",
        "a theorem is proven from axioms and previously proven theorems",
        "induction generalizes from specific observations to broad principles",
        "deduction derives specific conclusions from general principles",
        "abduction infers the most likely explanation for observations",
        "falsifiability means a claim can be proved wrong by evidence",
        "a paradigm is a framework of assumptions guiding inquiry",
        "paradigm shifts occur when evidence overwhelms the current framework",
        # Ethics
        "ethics studies what is right and wrong in human conduct",
        "consequentialism judges actions by their outcomes",
        "deontology judges actions by whether they follow moral rules",
        "virtue ethics focuses on the character of the moral agent",
        "the golden rule says treat others as you would like to be treated",
        "autonomy is the right of individuals to make their own decisions",
        "responsibility means being accountable for one's actions",
        "fairness means treating similar cases similarly",
        "transparency means actions and reasons are open to inspection",
        "the social contract is an implicit agreement to cooperate for mutual benefit",
        # Philosophy of mind
        "consciousness is subjective experience",
        "the hard problem of consciousness asks why physical processes produce experience",
        "qualia are the subjective qualities of conscious experience",
        "functionalism defines mind by what it does not what it is made of",
        "materialism holds that everything including mind is physical",
        "dualism holds that mind and body are separate substances",
        "emergence means complex properties arise from simpler components",
        "reductionism explains complex systems by analyzing their parts",
        "holism holds that systems have properties their parts do not",
        "intentionality is the mind's ability to represent or be about things",
        "free will is the capacity to choose between possible actions",
        "determinism holds that all events are caused by prior events",
        "compatibilism holds that free will and determinism are not contradictory",
        # Information and computation
        "information reduces uncertainty about the state of a system",
        "entropy in information theory measures uncertainty",
        "redundancy protects information against noise and errors",
        "compression removes redundancy to reduce data size",
        "a bit is the fundamental unit of information",
        "the church turing thesis says any computable function can be computed by a turing machine",
        "the halting problem is undecidable no algorithm can solve it for all programs",
        "computational complexity classifies problems by required resources",
        "NP problems have solutions that can be verified quickly",
        "P equals NP asks whether quickly verifiable problems are also quickly solvable",
        "godel incompleteness theorem says consistent formal systems cannot prove all truths",
        "an oracle provides answers to questions beyond a system's deductive reach",
        "self reference occurs when a system refers to itself",
        "strange loops arise from self referential hierarchical systems",
        "recursion in nature appears in fractals coastlines and branching trees",
        # Ontology and metaphysics
        "ontology studies what exists and the categories of being",
        "substance is that which persists through change",
        "properties are characteristics that things have",
        "relations connect two or more entities",
        "causation is the relationship between cause and effect",
        "necessity means something must be the case",
        "contingency means something could be otherwise",
        "identity asks what makes something the same over time",
        "the ship of theseus asks whether an object with all parts replaced is the same object",
        "time can be understood as a series of moments or as a continuous flow",
        "space provides the framework within which objects are located",
        "possible worlds are ways reality might have been",
        "abstract objects exist outside space and time",
        "concrete objects exist in space and time",
        "universals are properties shared by multiple particulars",
        "nominalism denies the existence of universals",
        "realism asserts that universals exist independently of particular things",
        # Philosophy of science
        "models simplify reality to make predictions",
        "idealization removes irrelevant details from a model",
        "explanation answers why questions about phenomena",
        "prediction uses theory to anticipate future observations",
        "theory testing compares predictions with observations",
        "a hypothesis is provisional until supported by strong evidence",
        "replicability strengthens confidence in scientific results",
        "operationalization defines abstract concepts through measurable procedures",
        "the demarcation problem asks what distinguishes science from non science",
        "scientific progress accumulates knowledge through observation and revision",
        # Philosophy of AI
        "the turing test evaluates whether a machine exhibits intelligent behavior",
        "the chinese room argument claims syntax alone is insufficient for understanding",
        "strong AI asserts that a sufficiently complex program has genuine understanding",
        "weak AI holds that machines simulate but do not genuinely think",
        "artificial general intelligence aims to match human cognitive flexibility",
        "alignment asks how to ensure AI systems pursue beneficial goals",
        "value alignment means an AI system's goals match human values",
        "instrumental convergence means many goals share certain subgoals",
        "interpretability enables humans to understand why an AI made a decision",
        "robustness means an AI system behaves correctly under new conditions",
        "a reward function specifies what an agent should optimize",
        "reward hacking occurs when an agent exploits loopholes in the reward function",
        "corrigibility means an AI system allows itself to be corrected",
        "an oracle AI answers questions without taking actions in the world",
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
    "aria":      generate_aria,
    "philosophy": generate_philosophy,
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
