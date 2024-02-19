import time
import asyncio


async def time_sleep():
    await asyncio.sleep(5)
    print("hellooooo")
    return "helo"


async def hello():
    x = await time_sleep()
    hello1()


def hello1():
    print("22222")


async def main():
    await hello()


if __name__ == '__main__':
    asyncio.run(main())
