# Stripe API Migration Guide

## Customer Creation

The legacy Customer API is no longer supported.

### Legacy API

```python
stripe.Customer.create(
    email="user@example.com"
)