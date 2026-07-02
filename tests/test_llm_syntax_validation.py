"""Tests for LLM Python code syntax validation."""

from agent.creative_writer import CreativeWriter, FALLBACK_CODE


class TestValidatePythonCode:
    """Test validate_python_code static method on CreativeWriter."""

    def test_invalid_llm_fragment(self):
        """A known bad fragment from LLM output should be detected as invalid."""
        bad = "#!/usr/bin/env python3\nto see if I am following good practices..."
        ok, err = CreativeWriter.validate_python_code(bad)
        assert not ok, f"Expected (False, ...), got ({ok}, {err!r})"
        assert err is not None, f"Expected an error message, got None"

    def test_valid_script(self):
        """A simple valid Python script should pass validation."""
        good = "#!/usr/bin/env python3\nprint('hello')\n"
        ok, err = CreativeWriter.validate_python_code(good)
        assert ok, f"Expected (True, None), got ({ok}, {err!r})"
        assert err is None, f"Expected None error, got {err!r}"

    def test_fallback_valid(self):
        """FALLBACK_CODE must itself be syntactically valid."""
        ok, err = CreativeWriter.validate_python_code(FALLBACK_CODE)
        assert ok, f"FALLBACK_CODE is invalid: {err}"
        assert err is None, f"Expected None error, got {err!r}"
