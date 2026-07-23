pub(crate) fn parse_seed_value(value: &str) -> Option<u64> {
    let value = value.trim();
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        if hex.is_empty() {
            return None;
        }
        u64::from_str_radix(hex, 16).ok()
    } else {
        value.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::parse_seed_value;

    #[test]
    fn parses_decimal_by_default_and_explicit_hex() {
        assert_eq!(parse_seed_value("12345"), Some(12_345));
        assert_eq!(parse_seed_value("0x12345"), Some(0x12345));
        assert_eq!(parse_seed_value("0XC0FFEE"), Some(0xC0FFEE));
    }

    #[test]
    fn rejects_invalid_or_empty_seed_values() {
        assert_eq!(parse_seed_value(""), None);
        assert_eq!(parse_seed_value("0x"), None);
        assert_eq!(parse_seed_value("not-a-seed"), None);
    }
}
