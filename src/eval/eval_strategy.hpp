#pragma once

#include "position/bbposition.hpp"
#include "core/types.hpp"

class IEvaluator {
public:
    virtual ~IEvaluator() = default;
    virtual int evaluate(const BitboardPosition& pos) const = 0;
};

class MaterialEvaluator final : public IEvaluator {
public:
    int evaluate(const BitboardPosition& pos) const override;
};

class NeuralEvaluator final : public IEvaluator {
public:
    int evaluate(const BitboardPosition& pos) const override;
};

class LassoEvaluator final : public IEvaluator {
public:
    int evaluate(const BitboardPosition& pos) const override;
};