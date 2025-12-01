## Getting Started

The package is tested and developed using python 3.12. While it may work with older and newer versions, this is the official recommendation.

It is highly recommended to work on this project using `uv` (check out the [installation guide](./installation.md) - it is very easy!). Using `uv` everything should work out of the box, if not feel free to create an issue!
The project includes two dependency groups depending on what you are working on. If you follow the installation guide using `uv`, the `--all-extras` flag will install all optional dependency groups automatically. The groups have the following scopes:

- `dev`: Testing and rapid prototyping using jupyter
- `docs`: Building the static and dynamic documentation (static markdown and docstrings)

## Development Workflow

This section contains an outline of how to work on the project.

### Naming branches

If you want to contribute directly in the repository create a new branch and start working. For branch names it is recommended to prefix the branch with the type of contribution you want to make. The branches are then named `<contribution-type>/<identifier>`. Here are some examples for contribution types:

- `feat`: If you want to add a new feature. An example could be `feat/env-sampling` for creating a feature where a user can sample an environment
- `refactor`: Refactoring existing code into a new format or structure. An example would be `refactor/formatters` if you want to change the interface of the `formatter` module.
- `fix`: For bugfixes, like `fix/mdp-abstraction`

This way the scope and goal of a branch is easy to determine.

### Writing Tests

When you work on the project it is highly recommended to start by [writing tests](#testing) especially if you are fixing a bug. Write a test that triggers the condition of the bug and fails because of it. After that go ahead and fix the bug and verify that your test case and all the others are passing. This way even if the bug reappears, it will be caught by the test suite.

The same goes for new features. If you specify the desired behavior of your code in a test first, you can quickly verify if you actually programmed what you wanted. Obviously this is just a recommendation, you can also prototype your code in a jupyter notebook and port it to the project afterwards (you can still write tests :wink:).
Every time you create a merge request, the whole test suite is run against the whole project, if any test fails you will get notified and make adjustments.

!!! info "Vectorized Environments"

    Every environment in the project is supposed to be vectorizable. Please always include a test for vectorizing environments whenever you implement a new type of environment.

### Formatting and Linting

The project is using [`ruff`](https://docs.astral.sh/ruff/) to format and lint code conforming to the PEP-standard.

## Coding Standards


When writing code please use [type hints](https://peps.python.org/pep-0484/) to specify the types of the parameters and return types. Most linters pick this up and make working with the project a lot easier.

### Docstrings

When writing docstrings use the [numpy](https://numpydoc.readthedocs.io/en/latest/format.html)-docstring format, which is automatically converted into the API-documentation format. If you create a new module and want to add it to the documentation you need to add a new markdown file (`.md`) to the `nav` section of the `mkdocs.yml` in the root directory of the project. To auto-generate the documentation based on docstrings add the following content to your file:

```md
::: verigym.<my_module>
```

### Naming conventions

For naming functions and classes please follow the [PEP-0008](https://peps.python.org/pep-0008/) naming conventions. Essentially this means:

- functions use snake case: `def my_func():`
- protected functions are prefixed with an underscore: `def _my_protected_func():`
- private functions are prefixed with two underscores: `def __my_protected_func():`
- classes are camel case: `class MyClass`

Generally it is safe to merge protected and private together since it tells users to not use this interface with confidence. However it allows them to use it, which is usually an advantage over something being strictly private.

### Project Structure

!!! question "Where should I add new code?"
    If you add code to an existing feature you should consider to put it into the same module (python file) as the code that already implements this feature.

    Otherwise create a new module with your additions. It is always possible to restructure code that is in the wrong location later so do not worry too much about it!

## Testing

The project uses [pytest](https://docs.pytest.org/en/stable/index.html) for testing. Pytest is a powerful testing framework that allows to write tests in a functional way. Using [fixtures](https://docs.pytest.org/en/stable/explanation/fixtures.html), you can inject data or mock objects into your tests that are automatically cleaned up after every test or test collection.


To execute the tests execute:

```sh
uv run pytest src
```

Which will print out all the collected tests and test results. Running all of the tests every single time you want to test your code can take a while. If you want to run only a single test file you can execute the test file directly:

```sh
uv run pytest tests/test_my_module.py
```

You can also run a single test function directly.

```sh
uv run pytest tests/test_my_module.py::test_my_function
```

## Documentation

The documentation is created using [mkdocs-material](https://squidfunk.github.io/mkdocs-material/). It is a powerful tool that creates beautiful documentation using only markdown! 

Material for mkdocs comes with a development server that you can use when you want to work on the documentation:

```sh
uv run mkdocs serve
```

Whenever you add new code make sure that you add [docstrings](#docstrings) to the new code that ideally contains (sorted by importance):

- **A short summary**
- **List of parameters, their types and return types**
- An example of how to use the function
- Errors that are possibly raised and why

The bold bullet points are **mandatory** as they are essential for users to comprehend what the function does and what it expects.

If you add a whole new feature consider writing a recipe, as it will give users a nice entrypoint when trying to use your feature!
