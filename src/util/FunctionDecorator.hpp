
#include <functional>
namespace dso {
  class FunctionDecorator
  {
  public:
    FunctionDecorator( std::function func )
      : m_func( func )

        void operator()()
    {
      // do some stuff prior to function call

      m_func();

      // do stuff after function call
    }

  private:
    std::function m_func;
  };

}
